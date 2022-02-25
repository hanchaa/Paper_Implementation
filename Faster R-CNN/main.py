import torch
from torch import nn

from anchor_utils import *
from faster_rcnn import FasterRCNN
from rpn import RPN
from vit_feature_extractor import ViTFeatureExtractor

device = "cpu" if torch.cuda.is_available() else "cpu"

faster_rcnn = FasterRCNN(ViTFeatureExtractor(3, 16, 768, 800, 12, 12, 4), RPN(768, 512, 9)).to(device)

rpn_cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
rpn_reg_loss_fn = nn.SmoothL1Loss()

dummy_bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]).to(device)
dummy_image = torch.zeros(1, 3, 800, 800).float().to(device)

if __name__ == "__main__":
    image_size = (800, 800)

    anchors = create_anchors([8, 16, 32], [0.5, 1, 2], 16, (50, 50), image_size)
    valid_anchor_indexes = get_valid_anchor_indexes(anchors, image_size)
    valid_anchors = anchors[valid_anchor_indexes]

    target_anchors_label, target_anchors_reg_parameter, ious = create_target_anchors(len(anchors), valid_anchors,
                                                                                     valid_anchor_indexes, dummy_bbox)

    predicted_anchors_reg_parameter, predicted_anchors_class_score = faster_rcnn(dummy_image)

    # loss of rpn
    gt_rpn_label = torch.from_numpy(target_anchors_label)
    rpn_score = predicted_anchors_class_score[0]
    rpn_cls_loss = rpn_cls_loss_fn(rpn_score, gt_rpn_label.long())

    mask = gt_rpn_label > 0
    gt_rpn_reg_parameter = torch.from_numpy(target_anchors_reg_parameter)[mask]
    rpn_reg_parameter = predicted_anchors_reg_parameter[0][mask]
    rpn_reg_loss = rpn_reg_loss_fn(rpn_reg_parameter, gt_rpn_reg_parameter)
