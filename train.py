# train_gol_unet.py
import os, random, json, math, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import UNetSmall  # your conv-UNet
from load_dataset import load_all_pairs, PATH  # your loader

# ----------------- Hyper-parameters -----------------
SEED = 0
LR = 1e-3
EPOCHS = 50  # raise if you like – early-stop handles it
BATCH_SIZE = 128
VAL_SPLIT = 0.20
PATIENCE = 3  # epochs with no val-improve before stop
MIN_DELTA = 1e-3
NUM_WORKERS = os.cpu_count() // 2
# ----------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin = device.type == "cuda"

# --------- Load (or build) dataset ----------
if os.path.exists("dataset.pt"):
    X, Y = torch.load("dataset.pt")
else:
    X, Y = load_all_pairs(PATH)  # ndarray → (N, H, W)
    X = torch.tensor(X[:, None], dtype=torch.float32)  # (N,1,H,W)
    Y = torch.tensor(Y[:, None], dtype=torch.float32)
    torch.save((X, Y), "dataset.pt")

# Sanity: ensure binary
X = (X > 0.5).float()
Y = (Y > 0.5).float()

ds = TensorDataset(X, Y)
train_len = int((1 - VAL_SPLIT) * len(ds))
val_len = len(ds) - train_len
train_ds, val_ds = random_split(
    ds, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin,
    num_workers=NUM_WORKERS,
)

# -------------- Model & Optimiser ---------------
model = UNetSmall().to(device)
criterion = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=2, verbose=True
)

# -------------- Training loop -------------------
best_val = math.inf
history = {"train": [], "val": []}
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    for xb, yb in tqdm(
        train_loader, leave=False, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch"
    ):
        xb, yb = xb.to(device), yb.to(device)

        optimiser.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimiser.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(train_loader.dataset)
    history["train"].append(train_loss)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in tqdm(
            val_loader, leave=False, desc=f"[{epoch}/{EPOCHS}] Val", unit="batch"
        ):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            val_loss += criterion(logits, yb).item() * xb.size(0)

    val_loss /= len(val_loader.dataset)
    history["val"].append(val_loss)

    # ---- Scheduler & Early-stop ----
    scheduler.step(val_loss)
    improved = val_loss + MIN_DELTA < best_val
    if improved:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "gol_unet_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping (no val improvement ≥{PATIENCE} epochs)")
            break

    tqdm.write(
        f"Epoch {epoch:02d} | "
        f"train {train_loss:.4f} | val {val_loss:.4f} "
        f"| lr {optimiser.param_groups[0]['lr']:.2e}"
    )

# Save final checkpoint
torch.save(model.state_dict(), "gol_unet_last.pt")
np.save("history.npy", history)
print(f"Training done.  Best val loss = {best_val:.4f}")

# -------------- Plot loss curves -----------------
plt.figure(figsize=(8, 5))
epochs = np.arange(1, len(history["train"]) + 1)
plt.plot(epochs, history["train"], label="Train loss")
plt.plot(epochs, history["val"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("BCEWithLogitsLoss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()
