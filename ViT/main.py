import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import nn, optim
from torch.utils.data import DataLoader

from early_stopping import EarlyStopping
from train import train, show_history
from vit import ViT


def show_sample_img(data, classes):
    indexes = np.random.randint(0, len(train_set), 4)

    x_grid = [data[i][0] for i in indexes]
    y_grid = [data[i][1] for i in indexes]

    x_grid = torchvision.utils.make_grid(x_grid, nrow=4, padding=2)
    plt.figure(figsize=(10, 10))

    np_img = x_grid.numpy()
    np_img_tr = np_img.transpose((1, 2, 0))

    plt.imshow(np_img_tr)

    plt.title(f"labels: {classes[y_grid[0]]} {classes[y_grid[1]]} {classes[y_grid[2]]} {classes[y_grid[3]]}")
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "../datasets"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    val_loader = DataLoader(test_set, batch_size=512, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    show_sample_img(train_set, classes)

    model = ViT(in_channels=3, patch_size=4, embedding_size=192, img_size=32, depth=12, num_heads=12, mlp_expansion=4,
                num_classes=10).to(device)

    model.load_state_dict(torch.load("./checkpoint.pt"))

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=0)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=15, warmup_steps=5, max_lr=0.007,
                                                 min_lr=0.00001, gamma=0.7)
    early_stopping = EarlyStopping(10, True)

    num_epochs = 100

    model, loss_history, metric_history = train(model, num_epochs, loss_fn, optimizer, train_loader, val_loader, device,
                                                lr_scheduler, early_stopping)

    show_history(num_epochs, loss_history, metric_history)
