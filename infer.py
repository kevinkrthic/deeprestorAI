import numpy as np
import torch
import matplotlib.pyplot as plt
from models.model import DeepRestorAI 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

model = DeepRestorAI().to(device)

model.load_state_dict(
    torch.load("deeprestorai.pth", map_location=device)
)

model.eval()

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


with torch.no_grad():
    restored = model(input_tensor)

restored = restored.squeeze().cpu().numpy()


np.save("restored.npy", restored)
print("Restored image saved as restored.npy")


print("Noisy Shape    :", noisy.shape)
print("Restored Shape :", restored.shape)
print("GT Shape       :", gt.shape)



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(restored, cmap="gray")
plt.title("Restored Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.show()