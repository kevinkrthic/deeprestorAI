import os
import time
from models.model import DeepRestorAI 
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


model = DeepRestorAI().to(device)

model.load_state_dict(
    torch.load("deeprestorai.pth", map_location=device)
)

model.eval()

lpips_model = lpips.LPIPS(net='alex').to(device)


image_name = "000000.npy"

noisy = np.load(
    f"dataset/train/NoisyLR/{image_name}"
).astype(np.float32)

gt = np.load(
    f"dataset/train/GT/{image_name}"
).astype(np.float32)


input_tensor = (
    torch.from_numpy(noisy)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.no_grad():
    output = model(input_tensor)

if device.type == "cuda":
    torch.cuda.synchronize()

end = time.perf_counter()

inference_time = (end - start) * 1000

restored = output.squeeze().cpu().numpy()


# ======================================================
# Metrics
# ======================================================

# MSE
mse = np.mean((gt - restored) ** 2)

# PSNR
psnr = peak_signal_noise_ratio(
    gt,
    restored,
    data_range=1.0
)

# SSIM
ssim = structural_similarity(
    gt,
    restored,
    data_range=1.0
)

# LPIPS
gt_tensor = (
    torch.from_numpy(gt)
    .unsqueeze(0)
    .unsqueeze(0)
)

pred_tensor = (
    torch.from_numpy(restored)
    .unsqueeze(0)
    .unsqueeze(0)
)

gt_tensor = gt_tensor.repeat(1, 3, 1, 1)
pred_tensor = pred_tensor.repeat(1, 3, 1, 1)

gt_tensor = gt_tensor * 2 - 1
pred_tensor = pred_tensor * 2 - 1

with torch.no_grad():
    lpips_score = lpips_model(
        pred_tensor.to(device),
        gt_tensor.to(device)
    ).item()


# ======================================================
# Model Statistics
# ======================================================
model_size = os.path.getsize("deeprestorai.pth") / (1024 * 1024)

parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


# ======================================================
# Results
# ======================================================
print("\n==============================")
print("Evaluation Results")
print("==============================")

print(f"MSE              : {mse:.6f}")
print(f"PSNR             : {psnr:.2f} dB")
print(f"SSIM             : {ssim:.4f}")
print(f"LPIPS            : {lpips_score:.4f}")
print(f"Inference Time   : {inference_time:.2f} ms")
print(f"Model Size       : {model_size:.2f} MB")
print(f"Parameters       : {parameters:,}")
print(f"Trainable Params : {trainable:,}")

print("==============================") 