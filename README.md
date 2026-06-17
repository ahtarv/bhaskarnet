# Bhaskar-NET: An Edge Deployment-First Network for Dense Object Detection

Official implementation of the paper **"Bhaskar NET - An edge deployment focused network, utilized for small object detection"**.

Bhaskar-NET is an edge-deployment-first single-stage object detector designed to bridge the "Inference Gap" between laboratory accuracy and field-deployable latency on CPU hardware. By replacing expensive deformable convolutions (which perform scattered memory reads via `grid_sample` causing a 0.1 FPS bottleneck on edge CPUs) with an optimized static convolution configuration and lightweight multi-scale neck fusion, Bhaskar-NET achieves a **35.5x CPU speedup** while retaining over **91% of state-of-the-art accuracy**.

---

## 🚀 Key Features
*   **TuaBottleneck & TuaAttention**: Custom Vision-Transformer (ViT) inspired attention blocks replacing standard YOLOv8 `C2f` blocks, employing LayerNorm and GELU activations.
*   **Hardware-Aware Approximations**: Replaces slow deformable convolutions with highly vectorizable static convolutions, fully utilizing AVX2/AVX-512 vectorization and caching.
*   **Lightweight neck fusion**: Novel `Scalseq` (Conv2D-based cross-scale sequence fusion) and `Zoomcat` (multi-zoom spatial aggregation) neck modules.
*   **AMP Compatible**: Fully compatible with Automatic Mixed Precision (FP16) training, saving memory and training time compared to deformable conv baselines.

---

## 📊 Performance Summary (VisDrone1k, 40 Epochs)

| Model | Parameters (M) | mAP50 | mAP50-95 | CPU Latency (ms) | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DFEM-Net (Baseline)** | 9.58 | **0.1450** | **0.0745** | ~9,736.0 | 0.1 |
| **BhaskarNet (Ours)** | **8.93** | **0.1350** | **0.0695** | **274.2** | **3.6** |
| **YOLOv8n** | 3.01 | 0.1230 | 0.0618 | ~310.5 | 3.2 |

*CPU Latency measured on consumer laptop grade CPU (Intel Core i3-1215U, 8GB RAM) at 800x800 resolution.*

---

## 🛠️ Repository Structure
```
BhaskarNet/
├── models/
│   ├── __init__.py           # Empty initialization file
│   ├── TuaAttention.py       # Custom spatial attention (with dilation toggle)
│   ├── TuaBottleneck.py      # Custom transformer bottleneck
│   ├── saclseq.py            # Scale-sequence fusion block
│   └── zoomcat.py            # Multi-zoom aggregation block
├── cfg/
│   └── bhaskar_net.yaml      # YOLOv8 model architecture configuration
├── train.py                  # Standalone training script (patches YOLOv8 & registers blocks)
├── val.py                    # Standalone evaluation/validation script
├── requirements.txt          # Dependencies (ultralytics, torch, etc.)
└── README.md                 # Documentation and Quick Start Guide
```

---

## 💻 Quick Start

### 1. Installation
Clone the repository and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets
Prepare your dataset (e.g. VisDrone, TinyPerson, or UAVDT) in standard YOLO format. Ensure you create a dataset configuration yaml file (e.g., `VisDrone.yaml`).

### 3. Training
To train Bhaskar-NET on your dataset:
```bash
python train.py --model cfg/bhaskar_net.yaml --data VisDrone.yaml --epochs 40 --imgsz 800 --batch 4 --name BhaskarNet_Run
```

### 4. Evaluation
To evaluate your trained model weights:
```bash
python val.py --weights BhaskarNet_Runs/BhaskarNet_Experiment/weights/best.pt --data VisDrone.yaml --imgsz 800
```

---

## 🧪 Ablation Study Insights
Our systematic ablation study of 8 variants highlights the following:
1.  **Standard Convolution vs. Dilated Convolution**: During ablations, standard convolutions (`dilation=1` inside `TuaAttention`) outperformed dilated convolutions (`dilation=2`) by **15.8%** at epoch 40 (0.132 vs 0.114 mAP50). This suggests that for dense, sub-pixel targets in aerial imagery, contiguous spatial locality is superior to wide-area dilation.
2.  **Skip Connections**: Removing residual skip connections in the TuaBottleneck blocks resulted in an **82.7% drop** in mAP50, confirming the absolute necessity of residual paths for Pre-LN structures.

---

## 📜 Citation
If you use Bhaskar-NET in your research, please cite our paper:
```bibtex
@article{patil2026bhaskarnet,
  title={Bhaskar NET - An edge deployment focused network, utilized for small object detection},
  author={Patil, Atharv Kamlesh and Varma, Satishkumar L.},
  journal={arXiv preprint},
  year={2026}
}
```
