import copy
import time

from loss import *


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def loss_batch(predicted_anchors_reg_parameter, predicted_anchors_class_score, bboxes, rpn_loss_fn, device,
               optimizer=None):
    loss = rpn_loss_fn(predicted_anchors_reg_parameter, predicted_anchors_class_score, bboxes, device)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def loss_epoch(model, dataloader, rpn_loss_fn, device, optimizer=None):
    running_loss = 0.0
    len_data = len(dataloader.dataset)
    idx = 1

    for img, bboxes, _ in dataloader:
        idx += 1
        img = img.to(device).float()
        bboxes = torch.Tensor(bboxes).to(device)

        predicted_anchors_location, predicted_anchors_class_score = model(img)

        loss = loss_batch(predicted_anchors_location, predicted_anchors_class_score, bboxes, rpn_loss_fn, device,
                          optimizer)
        print(f"{idx} / {len_data} / {loss}")

        running_loss += loss

    loss = running_loss / len_data

    return loss


def train(model, num_epochs, train_loader, validation_loader, rpn_loss_fn, optimizer, lr_scheduler, device):
    loss_history = {"train": [], "val": []}

    best_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)
        print(f"Epoch {epoch + 1}/{num_epochs}, current lr = {current_lr}")

        model.train()
        train_loss = loss_epoch(model, train_loader, rpn_loss_fn, device, optimizer)
        loss_history["train"].append(train_loss)

        # model.eval()
        # with torch.no_grad():
        #     val_loss = loss_epoch(model, validation_loader, device, valid_anchors, valid_anchor_indexes, num_anchors,
        #                           num_anchor_sample, rpn_lambda)
        # loss_history["val"].append(val_loss)
        #
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model_weights = copy.deepcopy(model.state_dict())
        #
        lr_scheduler.step(train_loss)

        print("train loss: %.6f / val loss: %.6f / time: %.4f min" % (
            train_loss, 0, (time.time() - start_time) / 60))
        print("-" * 10)

    model.load_state_dict(best_model_weights)

    return model, loss_history
