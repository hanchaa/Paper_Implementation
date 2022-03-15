import albumentations as A
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from fast_rcnn import FastRCNN
from faster_rcnn import FasterRCNN
from loss import *
from region_proposal import RegionProposal
from roi_sampler import RoiSampler
from rpn import RPN
from train import train
from vit_feature_extractor import ViTFeatureExtractor
from voc_dataset import VOCDataset, classes

colors = np.random.randint(0, 255, size=(20, 3), dtype="uint8")


def show(img, targets, labels, classes=classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)

    for target, label in zip(targets, labels):
        id = int(label)
        bbox = target[:4]

        color = [int(c) for c in colors[id]]
        name = classes[id]

        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255, 255, 255, 0))

    plt.imshow(np.array(img))
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "../datasets"

    train_dataset = VOCDataset(data_path, year="2007", image_set="train", download=True)
    validation_dataset = VOCDataset(data_path, year="2007", image_set="test", download=True)

    image_size = (224, 224)

    train_transforms = A.Compose([
        A.LongestMaxSize(max_size=image_size[0]),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[])
    )
    validation_transforms = A.Compose([
        A.LongestMaxSize(max_size=image_size[0]),
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT),
        ToTensorV2()
    ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=[])
    )

    train_dataset.transforms = train_transforms
    validation_dataset.transforms = validation_transforms

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    model = FasterRCNN(
        [2, 4, 8],
        [0.5, 1, 2],
        16,
        image_size,
        ViTFeatureExtractor(3, 16, 768, image_size[0], 12, 12, 4),
        RPN(768, 512, 9),
        RegionProposal(image_size, 12000, 2000, 6000, 300, 16, 0.7),
        RoiSampler(128, 0.25, 0.5, 0.5),
        FastRCNN((7, 7), 768, 16, 4096, 21),
        device
    ).to(device)

    rpn_loss_fn = RPNLoss(model.anchors, image_size, 256, 10)
    fast_rcnn_loss_fn = FastRCNNLoss(10)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    model, loss_history = train(model, 30, train_dataloader, validation_dataloader, rpn_loss_fn, fast_rcnn_loss_fn,
                                optimizer, lr_scheduler, device)
