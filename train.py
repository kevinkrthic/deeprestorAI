import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.dataset import SemiconductorDataset
from models.model import DeepRestoreAI

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 400
DEFAULT_LR = 2e-4


def build_loaders(train_root="dataset/train", val_root="dataset/val", batch_size=DEFAULT_BATCH_SIZE):
    if os.path.isdir(val_root) and os.path.isdir(os.path.join(val_root, "GT")) and os.path.isdir(os.path.join(val_root, "NoisyLR")):
        train_dataset = SemiconductorDataset(root_dir=train_root)
        val_dataset = SemiconductorDataset(root_dir=val_root)
    else:
        train_gt_dir = os.path.join(train_root, "GT")
        train_noisy_dir = os.path.join(train_root, "NoisyLR")

        if not os.path.isdir(train_gt_dir) or not os.path.isdir(train_noisy_dir):
            raise FileNotFoundError(
                f"Dataset folders not found. Expected '{train_gt_dir}' and '{train_noisy_dir}'"
            )

        all_files = sorted(os.listdir(train_gt_dir))
        if len(all_files) < 2:
            raise ValueError(f"Not enough training samples in {train_gt_dir}. Need at least 2 files.")

        val_count = max(1, len(all_files) // 10)
        val_files = all_files[:val_count]
        train_files = all_files[val_count:]

        train_dataset = SemiconductorDataset(root_dir=train_root, file_names=train_files)
        val_dataset = SemiconductorDataset(root_dir=train_root, file_names=val_files)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if device.type == "cpu" else 2
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


def main(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = build_loaders(batch_size=batch_size)
    model = DeepRestoreAI().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for noisy, gt in train_loader:
            noisy = noisy.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output = model(noisy)
                    loss = criterion(output, gt)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(noisy)
                loss = criterion(output, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy = noisy.to(device)
                gt = gt.to(device)

                output = model(noisy)
                loss = criterion(output, gt)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"""
Epoch [{epoch + 1}/{epochs}]
Train Loss : {train_loss:.6f}
Val Loss   : {val_loss:.6f}
LR         : {optimizer.param_groups[0]['lr']:.8f}
"""
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "deeprestoreai_v2_best.pth")
            print("Best Model Saved ✅")

    print("Training Completed")


if __name__ == "__main__":
    main()