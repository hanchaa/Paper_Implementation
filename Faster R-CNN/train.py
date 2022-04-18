import copy
import time

from torch.distributed import all_reduce
from tqdm import tqdm

from loss import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def loss_batch(output, bboxes, rpn_loss_fn, fast_rcnn_loss_fn, optimizer=None):
    loss = None
    if rpn_loss_fn is not None:
        predicted_anchors_reg_parameter, predicted_anchors_class_score = output[0], output[1]
        rpn_loss = rpn_loss_fn(predicted_anchors_reg_parameter, predicted_anchors_class_score, bboxes)
        loss = rpn_loss

    if fast_rcnn_loss_fn is not None:
        predicted_locations, predicted_class_scores, targets_location, targets_label = output[2], output[3], output[4], output[5]
        fast_rcnn_loss = fast_rcnn_loss_fn(predicted_locations, predicted_class_scores, targets_location, targets_label)

        if loss is None:
            loss = fast_rcnn_loss
        else:
            loss += fast_rcnn_loss

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def loss_epoch(model, dataloader, rpn_loss_fn, fast_rcnn_loss_fn, device, optimizer=None, verbose=False):
    running_loss = 0.0

    for img, bboxes, labels in tqdm(dataloader) if verbose else dataloader:
        img = img.to(device).float()
        bboxes = torch.Tensor(bboxes).to(device)

        output = model(img, bboxes, labels)

        loss = loss_batch(output, bboxes, rpn_loss_fn, fast_rcnn_loss_fn, optimizer)

        running_loss += loss

    return running_loss


def train(model, num_epochs, train_loader, validation_loader, optimizer, lr_scheduler, rpn_loss_fn=None,
          fast_rcnn_loss_fn=None,
          device="cpu", writer=None, verbose=False):
    best_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, current lr = {current_lr}")

        model.train()
        train_loss = loss_epoch(model, train_loader, rpn_loss_fn, fast_rcnn_loss_fn, device, optimizer, verbose=verbose)

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model, validation_loader, rpn_loss_fn, fast_rcnn_loss_fn, device, verbose=verbose)

        lr_scheduler.step()

        train_result = torch.tensor([train_loss, val_loss]).to(device)
        all_reduce(train_result)

        train_loss, val_loss = train_result
        train_loss /= len(train_loader.dataset)
        val_loss /= len(validation_loader.dataset)

        if verbose:
            print("train loss: %.6f / time: %.4f min" % (
                train_loss, (time.time() - start_time) / 60))
            print("-" * 10)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)

    return model
