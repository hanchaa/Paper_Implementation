import collections
import xml.etree.ElementTree as ET
from typing import Dict, Any

import numpy as np
from PIL import Image
from torchvision.datasets import VOCDetection
import torch
from torch.nn import functional as F

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    bboxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        bboxes.append(torch.tensor(b[1]))
        labels.append(b[2])

    images = torch.stack(images, dim=0).float()

    max_object_len = max([len(box) for box in bboxes])
    bboxes = [F.pad(bbox, pad=(0, 0, 0, max_object_len - len(bbox)), mode="constant", value=0) for bbox in bboxes]
    bboxes = torch.stack(bboxes, dim=0).float()

    for l in labels:
        for _ in range(max_object_len - len(l)):
            l.append(-1)

    return images, bboxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each


class VOCDataset(VOCDetection):
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())  # xml파일 분석하여 dict으로 받아오기

        targets = []  # 바운딩 박스 좌표
        labels = []  # 바운딩 박스 클래스

        # 바운딩 박스 정보 받아오기
        for t in target['annotation']['object']:
            label = np.zeros(5)
            label[:] = t['bndbox']['xmin'], t['bndbox']['ymin'], t['bndbox']['xmax'], t['bndbox'][
                'ymax'], classes.index(t['name'])

            targets.append(list(label[:4]))  # 바운딩 박스 좌표
            labels.append(int(label[4]))  # 바운딩 박스 클래스

        if self.transform:
            augmentations = self.transform(image=img, bboxes=targets)
            img = augmentations['image']
            targets = augmentations['bboxes']

        return img, targets, labels

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:  # xml 파일을 dictionary로 반환
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
