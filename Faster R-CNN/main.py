import albumentations as A
import cv2
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.distributed import init_process_group
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from faster_rcnn import FasterRCNN
from loss import *
from rpn import RPN
from train import train
from voc_dataset import VOCDataset


def main_worker(device, num_gpu):
    data_path = "./datasets"

    init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:3456",
        world_size=num_gpu,
        rank=device
    )
    torch.cuda.set_device(device)

    image_size = (800, 800)

    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size[0]),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[])
    )

    train_dataset = VOCDataset(data_path, year="2007", image_set="train", download=True, transforms=transform)
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=train_sampler)

    validation_dataset = VOCDataset(data_path, year="2007", image_set="val", download=True, transforms=transform)
    validation_sampler = DistributedSampler(validation_dataset)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, sampler=validation_sampler)

    vgg16 = torchvision.models.vgg16(pretrained=True)
    feature_extractor = vgg16.features[:30]

    model = FasterRCNN(
        [8, 16, 32],
        [0.5, 1, 2],
        16,
        image_size,
        feature_extractor,
        RPN(512, 512, 9),
        device=device
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    rpn_loss_fn = RPNLoss(model.module.anchors, image_size, 256, 10)
    # fast_rcnn_loss_fn = FastRCNNLoss(10)

    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, gamma=0.8, step_size=5)

    writer = SummaryWriter(comment=f"lr={lr}")

    model, loss_history = train(model, 30, train_dataloader, validation_dataloader, optimizer, lr_scheduler,
                                rpn_loss_fn, device=device, writer=writer, verbose=device == 0)

    if device == 0:
        torch.save(model.state_dict(), "./weights.pt")
        torch.save(model.module.feature_extractor.state_dict(), "./rpn_weight.pt")
        writer.flush()

    writer.close()


if __name__ == "__main__":
    num_gpu = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=num_gpu, args=(num_gpu,))
