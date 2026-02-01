# ConvMixer on Tiny ImageNet

This notebook demonstrates the training and evaluation of a **ConvMixer** model on the **Tiny ImageNet** dataset. The goal is to provide a clear, minimal, and reproducible baseline for convolution-only architectures that rely on aggressive spatial mixing rather than attention mechanisms.

---

## 1. Tiny ImageNet Dataset

### Overview

Tiny ImageNet is a reduced version of the ImageNet dataset designed for fast experimentation while retaining realistic visual complexity.

**Key properties:**

* **200 classes**
* **100,000 training images** (500 per class)
* **10,000 validation images** (50 per class)
* **Image resolution:** `64 × 64 × 3`

Official source:

```
http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

---

### Directory structure

After downloading and extracting, the dataset should have the following layout:

```
tiny-imagenet-200/
├── train/
│   ├── <class_id>/
│   │   ├── images/
│   │   └── *.txt
│   └── ... (200 classes)
├── val/
│   ├── images/
│   └── val_annotations.txt
├── test/
│   └── images/
├── wnids.txt
└── words.txt
```

* `wnids.txt` lists the class identifiers
* `words.txt` maps identifiers to human-readable labels
* Validation labels are provided via `val_annotations.txt`

---

## 2. ConvMixer Notebook

### Notebook: `ConvMixer.ipynb`

This notebook implements a **ConvMixer** architecture and trains it on Tiny ImageNet. ConvMixer is a convolution-only model that separates **spatial mixing** and **channel mixing** using depthwise and pointwise convolutions.

---

### Core architectural ideas

ConvMixer blocks follow a simple but effective pattern. I have adjusted the core architecture to achieve maximum performance while having few parameters
and speed.

1. **Conv Stem** via 5x5 strided convolution
2. **Depthwise convolution + Axial Depthwise convolution** for spatial mixing
3. **Pointwise (1×1) convolution** for channel mixing
4. **Residual connections** for stability

This design:

* Avoids attention entirely
* Preserves strong locality bias
* Scales well with depth
* Has 3 different stages

---

### High-level model structure

```
Input (64×64)
  ↓
Conv Stem 5x5 (Conv with stride 2)
  ↓
Stage 1: [ConvMixer Block (depthwise)] × N
  ↓
GroupConv 5x5 (Conv with stride 2)
  ↓
Stage 2: [ConvMixer Block (depthwise + axial depthwise)] × N
  ↓
GroupConv 5x5 (Conv with stride 2)
  ↓
Stage 3: [ConvMixer Block (axial depthwise)] × N
  ↓
Global Average Pooling
  ↓
Linear Classifier (200 classes)
```

Each ConvMixer block consists of:

* Depthwise and/or Axial depthwise convolution (large kernel)
* GELU activation
* Batch normalization
* Residual
* Pointwise convolution

---

## 3. Training setup

The notebook typically includes:

* Standard image preprocessing and normalization
* Data loading from Tiny ImageNet directory structure
* Cross-entropy loss for multi-class classification
* Adam or AdamW optimizer

Training Tiny ImageNet allows:

* Rapid iteration
* Clear comparison across architectural variants
* Evaluation of spatial inductive biases

---

## 4. Why ConvMixer + Tiny ImageNet?

This combination is useful because:

* ConvMixer benefits from **mid-scale spatial structure**
* Tiny ImageNet images are small enough to train quickly
* Results are more meaningful than CIFAR but cheaper than ImageNet

The notebook serves as a **baseline** for later architectural extensions (e.g. spatial shuffling, patch unfolding, wavelet mixing).

---

## 5. Running the notebook

### Requirements

* Python 3.9+
* TensorFlow 2.x
* NumPy
* Matplotlib

Optional:

* GPU for faster training

---

### Dataset setup

1. Download Tiny ImageNet:

   ```bash
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   ```

2. Ensure the dataset path in the notebook matches your local setup.

---

## 6. Scope and limitations

This notebook is intended to:

* Provide a clean adapted ConvMixer reference implementation
* Serve as a comparison point for more experimental models
* I have not found a model with fewer parameters (2.6M) having higher validation accuarcy (~63%) than this (01.02.2026).

---

## 7. Possible extensions

* Replace patch embedding with lossless spatial shuffling
* Compare against pooling-based CNNs
* Analyze kernel size vs performance
* Extend to dense prediction tasks

---

## 8. Attribution

* Tiny ImageNet dataset: Stanford CS231n
* ConvMixer architecture: Trockman & Kolter (2022)

---

**This notebook is best viewed as a strong convolutional baseline and a stepping stone toward more structured spatial mixing architectures.**
