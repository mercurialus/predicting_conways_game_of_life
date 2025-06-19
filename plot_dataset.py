import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt

if os.path.exists("dataset.pt"):
    X, Y = torch.load("dataset.pt")

dataset = TensorDataset(X, Y)
num_samples_to_plot = 1

for i in range(num_samples_to_plot):
    x_frame = dataset[i][0]
    y_frame = dataset[i][1]

    # Reshape if needed (assuming square images, e.g., 28x28)
    img_size = int(x_frame.numel() ** 0.5)
    x_img = x_frame.view(img_size, img_size)
    y_img = y_frame.view(img_size, img_size)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(x_img, cmap="gray")
    axes[0].set_title(f"Frame t (sample {i})")
    axes[0].axis("off")

    axes[1].imshow(y_img, cmap="gray")
    axes[1].set_title("Frame t+1")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
