# DeepRestorAI

DeepRestorAI is a lightweight PyTorch image restoration model designed to remove noise and recover fine details from degraded grayscale semiconductor / inspection-like images. The pipeline trains on paired noisy and clean image samples and predicts a restored output close to the ground truth.

## Project Objective

This project aims to solve a low-quality image restoration problem where each noisy input image is paired with a clean target image. The model learns a mapping from noisy low-resolution input to a denoised, clearer output.

The project is built around a custom encoder-decoder CNN with residual attention blocks and pixel-shuffle upscaling for reconstruction.

## Problem Statement

The dataset contains:

- NoisyLR: degraded/noisy low-quality input images
- GT: clean ground-truth reference images

The task is to restore the noisy input so that it visually matches the GT as closely as possible.

## Model Architecture

The model in [models/model.py](models/model.py) follows a custom deep restoration design:

- Encoder blocks for feature extraction
- Downsampling layers for hierarchical context learning
- Residual attention blocks for better feature refinement
- Decoder blocks with skip connections
- Upscaling layer to reconstruct final image dimensions
- Final output layer producing the restored grayscale image

This architecture is optimized for image enhancement tasks where preserving structure and edge detail is important.

## Dataset Structure

```text
dataset/
├── train/
│   ├── GT/
│   └── NoisyLR/
└── val/   # optional validation split if present
    ├── GT/
    └── NoisyLR/
```

Each file is stored as a `.npy` grayscale array, and the dataloader pairs matching noisy and ground-truth samples.

## Repository Structure

```text
KLA_DeepRestorAI/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── evaluate.py
├── utils.py
├── deeprestorai.pth
├── deeprestoreai_best.pth
├── dataset/
│   ├── dataset.py
│   └── train/
│       ├── GT/
│       └── NoisyLR/
├── models/
│   ├── __init__.py
│   └── model.py
└── assets/
    ├── result_comparison.png
    └── evaluation_summary.png
```

## Training Pipeline

Training is handled by [train.py](train.py). The script:

1. loads the dataset
2. builds the data loader
3. initializes the model
4. defines L1 loss and AdamW optimizer
5. trains for multiple epochs
6. validates on a held-out validation set
7. saves the best model checkpoint

### Training command

```bash
python train.py
```

## Inference Pipeline

Inference is handled by [infer.py](infer.py). It:

1. loads the trained model checkpoint
2. reads a noisy input file
3. predicts the restored image
4. saves the output to `restored.npy`
5. displays noisy, restored, and GT images side by side

### Inference command

```bash
python infer.py
```

## Evaluation Pipeline

The evaluation script [evaluate.py](evaluate.py) calculates:

- MSE
- PSNR
- SSIM
- LPIPS
- inference time
- model size
- parameter count

This helps quantify reconstruction quality and compare model performance against ground truth.

### Evaluation command

```bash
python evaluate.py
```

## Sample Results

The restored output is visually close to the ground truth and shows improved detail recovery compared to the noisy input.

<img src="assets/result_comparison.png" alt="Noisy, Restored, and Ground Truth comparison" width="100%">

<img src="assets/evaluation_summary.png" alt="Evaluation metrics summary" width="80%">

## Example Metrics

The model achieved the following output on a sample evaluation run:

- MSE: 0.000583
- PSNR: 32.35 dB
- SSIM: 0.9035
- LPIPS: 0.1342
- Inference Time: 298.09 ms
- Model Size: 2.41 MB
- Parameters: 624,725

These values indicate strong restoration quality and competitive reconstruction performance on the test sample.

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

The project uses:

- torch
- torchvision
- opencv-python
- numpy
- matplotlib
- tqdm
- Pillow
- scikit-image
- lpips
- albumentations

## How to Use

### 1. Prepare dataset
Place your paired noisy and GT images in:

- `dataset/train/NoisyLR/`
- `dataset/train/GT/`

### 2. Train model

```bash
python train.py
```

### 3. Run inference

```bash
python infer.py
```

### 4. Evaluate

```bash
python evaluate.py
```

## Notes

- The repository keeps the trained checkpoint files (`.pth`) so the model can be reused immediately.
- The project is suited for hackathon demos and image restoration experiments.
- If you want to improve the model further, consider larger training data, better augmentation, or deeper restoration blocks.

## License

This project is intended for academic and hackathon demo use.

## Acknowledgement

This work is a custom deep learning restoration pipeline built for denoising and reconstruction of degraded image pairs.
"# DeepRestor.ai" 
"# DeepRestor.ai" 
