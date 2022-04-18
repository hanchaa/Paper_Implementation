import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from early_stopping import EarlyStopping
from train import train
from vit import ViT


def main_worker(device, num_gpus_per_node):
    data_path = "./datasets"

    train_batch_size = 100 // num_gpus_per_node
    test_batch_size = 256 // num_gpus_per_node

    init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:3456",
        world_size=num_gpus_per_node,
        rank=device
    )
    torch.cuda.set_device(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, pin_memory=True, num_workers=4, shuffle=False,
                              sampler=train_sampler)

    val_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    val_sampler = DistributedSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=test_batch_size, pin_memory=True, num_workers=4, shuffle=False,
                            sampler=val_sampler)

    dropout = 0.1
    model = ViT(in_channels=3, patch_size=16, embedding_size=768, img_size=224, depth=12, num_heads=12, mlp_expansion=4,
                num_classes=100, patch_embedding_dropout=dropout, attention_dropout=dropout, mlp_dropout=dropout).to(device)

    # model.load_state_dict(torch.load("./checkpoint.pt"))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    lr = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    early_stopping = EarlyStopping(10, True)

    num_epochs = 100
    writer = SummaryWriter(comment=f"lr={lr}, dropout={dropout}")

    model = train(model, num_epochs, loss_fn, optimizer, train_loader, val_loader, device, lr_scheduler, early_stopping,
                  writer, verbose=device == 0)

    if device == 0:
        torch.save(model.state_dict(), "./weights.pt")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    num_gpus_per_node = torch.cuda.device_count()

    torch.multiprocessing.spawn(main_worker, nprocs=num_gpus_per_node, args=(num_gpus_per_node,))
