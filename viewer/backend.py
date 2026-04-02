import re
import sys
import json
import io
import base64
import asyncio
import threading
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add LP-Diff root so we can import model/core/data packages
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

app = FastAPI()

EXPERIMENTS_DIR = _ROOT / "experiments"
STATIC_DIR = Path(__file__).parent / "static"
CONFIG_PATH = _ROOT / "config" / "LP-Diff.json"

# Serve experiment images
app.mount("/experiments", StaticFiles(directory=str(EXPERIMENTS_DIR)), name="experiments")

# Serve frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _has_results(exp_dir: Path) -> bool:
    results = exp_dir / "results"
    if not results.is_dir():
        return False
    return any(
        f.suffix == ".png"
        for epoch_dir in results.iterdir()
        if epoch_dir.is_dir()
        for f in epoch_dir.iterdir()
    )


@app.get("/api/experiments")
def list_experiments():
    if not EXPERIMENTS_DIR.exists():
        return {"experiments": []}
    runs = sorted(
        [d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and _has_results(d)],
        reverse=True,
    )
    return {"experiments": runs}


@app.get("/api/experiments/{exp}/info")
def experiment_info(exp: str):
    exp_dir = EXPERIMENTS_DIR / exp
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    results_dir = exp_dir / "results"
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results directory")

    # Parse all epoch directories
    data = {}  # epoch_str -> {plate_id -> {iters, hr, sr, lr1, lr2, lr3}}
    all_plates = set()

    FILE_RE = re.compile(r"^(\d+)_(\d+)_(hr|sr|lr1|lr2|lr3)\.png$")

    for epoch_dir in results_dir.iterdir():
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name)
        except ValueError:
            continue

        epoch_data = {}
        for f in epoch_dir.iterdir():
            m = FILE_RE.match(f.name)
            if not m:
                continue
            iters, plate_id, img_type = m.group(1), m.group(2), m.group(3)
            if plate_id not in epoch_data:
                epoch_data[plate_id] = {"iters": int(iters)}
            epoch_data[plate_id][img_type] = f.name

        if epoch_data:
            data[str(epoch)] = epoch_data
            all_plates.update(epoch_data.keys())

    if not data:
        raise HTTPException(status_code=404, detail="No result images found")

    epochs = sorted(data.keys(), key=lambda x: int(x))
    plates = sorted(all_plates, key=lambda x: int(x))

    return {"epochs": epochs, "plates": plates, "data": data}


# ── Log regexes ───────────────────────────────────────────────────────────────
_RE_TRAIN_AVG = re.compile(
    r"<epoch:\s*(\d+), iter:\s*([\d,]+)>\s+avg_train_loss:\s*([\d.e+\-]+)"
)
_RE_TRAIN_DETAIL = re.compile(
    r"<epoch:\s*(\d+), iter:\s*([\d,]+)>\s+"
    r"l_pix:\s*([\d.e+\-]+)\s+l_diffusion:\s*([\d.e+\-]+)\s+l_mta:\s*([\d.e+\-]+)\s+lr:\s*([\d.e+\-]+)"
)
_RE_VAL = re.compile(
    r"<epoch:\s*(\d+), iter:\s*([\d,]+)>\s+psnr:\s*([\d.e+\-]+)\s+loss:\s*([\d.e+\-]+)"
)


def _parse_log(path: Path, regex) -> list:
    if not path.exists():
        return []
    results = []
    for line in path.read_text(errors="ignore").splitlines():
        m = regex.search(line)
        if m:
            results.append(m.groups())
    return results


@app.get("/api/experiments/{exp}/metrics")
def experiment_metrics(exp: str):
    exp_dir = EXPERIMENTS_DIR / exp
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    logs_dir = exp_dir / "logs"
    train_log = logs_dir / "train.log"
    val_log = logs_dir / "val.log"

    # avg_train_loss per epoch
    train = [
        {"epoch": int(e), "iter": int(i.replace(",", "")), "avg_train_loss": float(v)}
        for e, i, v in _parse_log(train_log, _RE_TRAIN_AVG)
    ]

    # detailed per-batch losses
    train_detail = [
        {
            "epoch": int(e), "iter": int(i.replace(",", "")),
            "l_pix": float(lp), "l_diffusion": float(ld),
            "l_mta": float(lm), "lr": float(lr),
        }
        for e, i, lp, ld, lm, lr in _parse_log(train_log, _RE_TRAIN_DETAIL)
    ]

    # validation metrics
    val = [
        {"epoch": int(e), "iter": int(i.replace(",", "")), "psnr": float(p), "loss": float(l)}
        for e, i, p, l in _parse_log(val_log, _RE_VAL)
    ]

    return {"train": train, "train_detail": train_detail, "val": val}


# ── Inference ─────────────────────────────────────────────────────────────────

_model_cache: dict = {}   # exp_name -> {"netG": ..., "device": ...}
_model_lock = threading.Lock()


def _strip_json_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", text)


def _parse_config() -> dict:
    raw = CONFIG_PATH.read_text()
    return json.loads(_strip_json_comments(raw))


def _find_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt_dir = exp_dir / "checkpoint"
    if not ckpt_dir.exists():
        return None

    def _iter_num(p: Path) -> int:
        m = re.match(r"I(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    # Priority: best combined > best psnr > best loss > latest regular
    for suffix in ("_gen_best.pth", "_gen_best_psnr.pth", "_gen_best_loss.pth"):
        candidates = sorted(ckpt_dir.glob(f"*{suffix}"), key=_iter_num)
        if candidates:
            return candidates[-1]
    candidates = sorted(ckpt_dir.glob("*_gen.pth"), key=_iter_num)
    if candidates:
        return candidates[-1]
    return None


@app.get("/api/infer/experiments")
def list_infer_experiments():
    """List all experiments that have a checkpoint directory."""
    if not EXPERIMENTS_DIR.exists():
        return {"experiments": []}
    runs = []
    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if d.is_dir() and _find_checkpoint(d) is not None:
            runs.append(d.name)
    return {"experiments": runs}


@app.get("/api/infer/status/{exp}")
def infer_status(exp: str):
    exp_dir = EXPERIMENTS_DIR / exp
    ckpt = _find_checkpoint(exp_dir)
    loaded = exp in _model_cache
    return {
        "loaded": loaded,
        "has_checkpoint": ckpt is not None,
        "checkpoint": ckpt.name if ckpt else None,
    }


def _load_model_sync(exp_name: str):
    """Load model weights into cache (blocking, runs in thread pool)."""
    import torch
    import model.networks as networks

    exp_dir = EXPERIMENTS_DIR / exp_name
    ckpt_file = _find_checkpoint(exp_dir)
    if ckpt_file is None:
        raise FileNotFoundError(f"No checkpoint found in {exp_dir / 'checkpoint'}/")

    config = _parse_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt = {
        "phase": "val",
        "gpu_ids": [0] if torch.cuda.is_available() else None,
        "local_rank": 0,
        "distributed": False,
        "model": config["model"],
        "train": {
            "lambda_mta": config.get("train", {}).get("lambda_mta", 1.0),
            "use_prerain_MTA": False,
            "resume_training": False,
        },
        "path": {"resume_state": None},
    }

    # Build architecture (no weight init since phase='val')
    netG = networks.define_G(opt)

    # Init schedule buffers first (needed so state_dict keys match checkpoint)
    train_schedule = config["model"]["beta_schedule"]["train"]
    netG.set_new_noise_schedule(train_schedule, device)

    # Load weights
    state_dict = torch.load(str(ckpt_file), map_location=device, weights_only=True)
    netG.load_state_dict(state_dict, strict=True)

    # Switch to val schedule for inference
    val_schedule = config["model"]["beta_schedule"]["val"]
    netG.set_new_noise_schedule(val_schedule, device)

    netG.to(device)
    netG.eval()

    logging.getLogger(__name__).info(f"Loaded model for '{exp_name}' from {ckpt_file.name}")

    with _model_lock:
        _model_cache[exp_name] = {"netG": netG, "device": device}


async def _get_or_load_model(exp_name: str) -> dict:
    with _model_lock:
        if exp_name in _model_cache:
            return _model_cache[exp_name]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model_sync, exp_name)
    return _model_cache[exp_name]


def _preprocess_images(image_bytes_list: list):
    """
    Decode images, resize to (112, 224), normalize to [-1, 1].
    Pads to 3 frames by repeating the last one.
    Returns tensor (1, 3, 3, 112, 224).
    """
    import torch
    import numpy as np
    from PIL import Image

    tensors = []
    for img_bytes in image_bytes_list:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 112))  # PIL uses (W, H)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # → [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 112, 224)
        tensors.append(t)

    while len(tensors) < 3:
        tensors.append(tensors[-1].clone())

    return torch.stack(tensors, dim=0).unsqueeze(0)  # (1, 3, 3, 112, 224)


def _tensor_to_png_b64(tensor) -> str:
    """Convert (3, H, W) tensor in [-1, 1] to base64-encoded PNG string."""
    import numpy as np
    from PIL import Image

    arr = tensor.detach().float().cpu().numpy()
    arr = (arr * 0.5 + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8).transpose(1, 2, 0)  # (H, W, 3)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/api/infer/{exp}")
async def run_inference(exp: str, files: List[UploadFile] = File(...)):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Provide at least 1 image")
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 images — got {}".format(len(files)))

    exp_dir = EXPERIMENTS_DIR / exp
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    image_bytes = [await f.read() for f in files]

    try:
        ctx = await _get_or_load_model(exp)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    netG = ctx["netG"]
    device = ctx["device"]

    import torch

    try:
        lr_seq = _preprocess_images(image_bytes).to(device)  # (1, 3, 3, 112, 224)
        netG.eval()
        with torch.no_grad():
            condition = netG.MTA(*lr_seq.unbind(1))          # (1, 3, 112, 224)
            sr = netG.super_resolution(condition, continuous=False)  # (1, 3, 112, 224)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "sr": _tensor_to_png_b64(sr[0]),
        "condition": _tensor_to_png_b64(condition[0]),
        "lr_inputs": [_tensor_to_png_b64(lr_seq[0, i]) for i in range(len(image_bytes))],
        "n_frames": len(image_bytes),
    }
