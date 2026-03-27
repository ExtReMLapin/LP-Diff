import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
STATIC_DIR = Path(__file__).parent / "static"

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
