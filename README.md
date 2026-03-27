<p align="center">
  <a href="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/results_with_word.png">
    <img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/results_with_word.png" width="80%">
  </a>
</p>

<h1 align="center">LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate</h1>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Gong_LP-Diff_Towards_Improved_Restoration_of_Real-World_Degraded_License_Plate_CVPR_2025_paper.html"><img src="https://img.shields.io/badge/Paper-CVPR%202025-blue.svg"></a>
</p>

<p align="center">
  <b>Haoyan Gong</b><sup>1</sup>, <b>Zhenrong Zhang</b><sup>1</sup>, <b>Yuzheng Feng</b><sup>1</sup>, <b>Anh Nguyen</b><sup>2</sup>, <b>Hongbin Liu*</b><sup>1</sup><br>
  <sup>1</sup>Xi’an Jiaotong-Liverpool University, <sup>2</sup>University of Liverpool <br>
  <a href="mailto:m.g.haoyan@gmail.com">Contact: m.g.haoyan@gmail.com</a>
</p>

---

## 📝 Abstract

License plate (LP) recognition is crucial for intelligent traffic management. Real-world LP images are often severely degraded due to distance and camera quality, making restoration extremely challenging.  
We introduce the first real-world multi-frame paired LP restoration dataset (**MDLP**, 11,006 groups) and a diffusion-based restoration model LP-Diff featuring: Inter-frame Cross Attention for multi-frame fusion; Texture Enhancement for recovering fine details; Dual-Pathway Fusion for effective channel/spatial selection
Our method **outperforms prior SOTA** on real LP images, both quantitatively and visually.

---

## 🔥 Highlights

- **[MDLP Dataset]**: First real-world, paired, multi-frame LP restoration dataset (11,006 groups).
    
- **[Diffusion-based Model]**: Custom architecture tailored for license plate restoration.
    
- **[SOTA Performance]**: Best on MDLP for both image quality and LP recognition.
    

---

## 🌟 Visual Results

**Qualitative comparison on real-world LP images:**

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/results_v2.png" width="100%"/>

**These are some confusing letters and complex Chinese characters:**

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/char_chinese.png" width="100%"/>


## 📊 Quantitative Results

|   Method    |  PSNR ↑   |  SSIM ↑   |  FID ↓   |  LPIPS ↓  |   NED ↓   |   ACC ↑   |
| :---------: | :-------: | :-------: | :------: | :-------: | :-------: | :-------: |
|    SRCNN    |   14.01   |   0.195   |  248.3   |   0.517   |   0.626   |   0.041   |
|     HAT     |   14.16   |   0.250   |  229.6   |   0.413   |   0.613   |   0.050   |
| Real-ESRGAN |   13.93   |   0.369   |   31.0   |   0.176   |   0.279   |   0.161   |
|   ResDiff   |   12.00   |   0.269   |   35.9   |   0.277   |   0.292   |   0.159   |
|  ResShift   |   12.53   |   0.321   |   89.1   |   0.288   |   0.332   |   0.099   |
| **LP-Diff** | **14.40** | **0.393** | **22.0** | **0.159** | **0.198** | **0.305** |

_(On MDLP real-world test set. NED: normalized edit distance; ACC: text recognition accuracy)_

---

## 🏗️ Model Overview

- **ICAM**: Inter-frame Cross Attention Module
    
- **TEM**: Texture Enhancement Module
    
- **DFM**: Dual-Pathway Fusion Module
    
- **RCDM**: Residual Condition Diffusion Module

 <img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/backbone_v2.png" width="100%"/>

---

## 📚 Dataset

The **MDLP Dataset** consists of 11,006 groups of real-world degraded license plate images. The dataset was collected under diverse real-world conditions, including various distances, illumination changes, and weather conditions. It provides multi-frame degraded images with corresponding clear ground-truth images for robust restoration model training.

**Dataset collection pipeline:**

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/data collection.png" width="100%"/>

**Example images:**

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/dataset.png" width="100%"/>

**Detail of one license plate image:**

<img src="https://raw.githubusercontent.com/haoyGONG/LP-Diff/main/figs/dataset_onedetail.png" width="100%"/>


---

## 🚀 Getting Started

### 1. Installation

- Install Python and required dependencies.
    
- Then install remaining Python packages:
    
    ```bash
    pip install -r requirements.txt
    ```
    

### 2. Download MDLP Dataset

- [Google Drive](https://drive.google.com/file/d/1UpECGcWcF92z-P6pJ9couzGTXb1TMHqk/view?usp=sharing)
    
- [Baidu Netdisk (access code: 1ebm)](https://pan.baidu.com/s/1Aphb_jIx_0tRR71BBbwVwA?pwd=1ebm)
    

### 3. Training & Evaluation

- **Training (single GPU):**

    ```bash
    python run.py -p train -c ./config/LP-Diff.json -gpu 0
    ```

- **Training (multi-GPU with DDP):**

    ```bash
    torchrun --nproc_per_node=<NUM_GPUS> run.py -p train -c ./config/LP-Diff.json -gpu 0,1,...
    ```

    Example with 2 GPUs:

    ```bash
    torchrun --nproc_per_node=2 run.py -p train -c ./config/LP-Diff.json -gpu 0,1
    ```

- **Validation:**

    ```bash
    python run.py -p val -c ./config/LP-Diff.json -gpu 0
    ```

- Results and checkpoints are saved in `./experiments`. When using multi-GPU training, logging and checkpointing are performed exclusively by rank 0.
    

---

## 🖼️ Experiment Viewer

A local web viewer to explore training results across epochs is available in `viewer/`.

<img width="3839" height="1330" alt="image" src="https://github.com/user-attachments/assets/b7fb07b0-3841-4a46-a441-d633fabb4566" />


**Features:**
- Select any experiment run from the dropdown (only runs with results are shown)
- Epoch slider to browse super-resolution quality over training iterations
- Prefetch of adjacent epochs for smooth navigation
- Click any plate to compare HR / LR1 / LR2 / LR3 / SR side by side

**Setup (one-time):**

```bash
cd viewer
pip install -r requirements.txt
```

**Launch:**

```bash
cd viewer
uvicorn backend:app --port 8765
```

Then open `http://localhost:8765` in your browser.

---

## 📂 Project Structure

```
LP-Diff/
│
├── config/              # Training and testing config files
├── data/                # Data loading scripts
├── experiments/         # Model checkpoints and logs
├── figs/                # Visualization images for README and paper
├── models/              # Model implementations
├── viewer/              # Local web viewer for experiment results
├── requirements.txt     # Python dependencies
└── run.py               # Main training/testing script
```

---

## 📖 Citation

If you use this work or dataset, please cite:

```bibtex
@inproceedings{gong2025lp,
  title={LP-Diff: Towards Improved Restoration of Real-World Degraded License Plate},
  author={Gong, Haoyan and Zhang, Zhenrong and Feng, Yuzheng and Nguyen, Anh and Liu, Hongbin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17831--17840},
  year={2025}
}
```

---

## 🤝 Acknowledgements

This project is based on the excellent [ResDiff](https://github.com/LYL1015/ResDiff/tree/master) codebase.  
We gratefully acknowledge all related open-source works.

---

## 💬 Contact

For questions, open an issue or email:  
**[m.g.haoyan@gmail.com](mailto:m.g.haoyan@gmail.com)**

---
