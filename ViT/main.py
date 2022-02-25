import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train import train, show_history
from vit import ViT


def show_sample_img(data):
    indexes = np.random.randint(0, len(train_set), 4)

    x_grid = [data[i][0] for i in indexes]
    y_grid = [data[i][1] for i in indexes]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=4, padding=2)
    plt.figure(figsize=(10, 10))

    np_img = x_grid.numpy()
    np_img_tr = np_img.transpose((1, 2, 0))

    plt.imshow(np_img_tr)

    plt.title("labels: " + str(y_grid))
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "../datasets"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)])

    train_set = torchvision.datasets.STL10(root=data_path, split="train", download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    test_set = torchvision.datasets.STL10(root=data_path, split="test", download=True, transform=transform)
    val_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    show_sample_img(train_set)

    model = ViT(in_channels=3, patch_size=16, embedding_size=768, img_size=224, depth=12, num_heads=12, mlp_expansion=4,
                num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    num_epochs = 100

    model, loss_history, metric_history = train(model, num_epochs, loss_fn, optimizer, train_loader, val_loader, device,
                                                lr_scheduler)

    show_history(num_epochs, loss_history, metric_history)
