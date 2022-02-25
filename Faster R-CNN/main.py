from faster_rcnn import FasterRCNN
from loss import *
from rpn import RPN
from vit_feature_extractor import ViTFeatureExtractor

device = "cpu" if torch.cuda.is_available() else "cpu"

faster_rcnn = FasterRCNN(ViTFeatureExtractor(3, 16, 768, 800, 12, 12, 4), RPN(768, 512, 9)).to(device)

dummy_bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]).to(device)
dummy_image = torch.zeros(1, 3, 800, 800).float().to(device)

if __name__ == "__main__":
    image_size = (800, 800)

    anchors = create_anchors([8, 16, 32], [0.5, 1, 2], 16, (50, 50), image_size)
    valid_anchor_indexes = get_valid_anchor_indexes(anchors, image_size)
    valid_anchors = anchors[valid_anchor_indexes]

    num_anchor_sample = 256
    rpn_lambda = 10

    predicted_anchors_reg_parameter, predicted_anchors_class_score = faster_rcnn(dummy_image)

    rpn_loss, ious = calc_rpn_loss(predicted_anchors_class_score, predicted_anchors_reg_parameter, valid_anchors,
                                   valid_anchor_indexes, dummy_bbox, len(anchors), num_anchor_sample,
                                   rpn_lambda)

    print(rpn_loss)
