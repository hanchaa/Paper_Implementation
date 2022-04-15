import copy
import time

import matplotlib.pyplot as plt
import torch
from torch.distributed import all_reduce
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def metric_batch(output, target):
    predicted = output.argmax(-1)
    corrects = predicted.eq(target).sum().item()
    return corrects


def loss_batch(loss_fn, output, target, optimizer=None):
    loss = loss_fn(output, target)
    metric = metric_batch(output, target)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), metric


def loss_epoch(model, loss_fn, dataloader, device, optimizer=None, verbose=False):
    running_loss = 0.0
    running_metric = 0.0

    for x_batch, y_batch in tqdm(dataloader) if verbose else dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)

        loss, metric = loss_batch(loss_fn, output, y_batch, optimizer)

        running_loss += loss
        running_metric += metric

    loss = running_loss
    metric = running_metric

    return loss, metric


def train(model, num_epochs, loss_fn, optimizer, train_loader, validation_loader, device, lr_scheduler, early_stopping,
          verbose=False):
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}

    best_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, current lr = {current_lr}")

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_fn, train_loader, device, optimizer, verbose)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_fn, validation_loader, device, verbose=verbose)

        lr_scheduler.step()

        train_result = torch.tensor([train_loss, train_metric, val_loss, val_metric]).to(device)
        all_reduce(train_result)

        train_loss, train_metric, val_loss, val_metric = train_result.cpu()
        train_loss /= len(train_loader.dataset)
        train_metric /= len(train_loader.dataset)
        val_loss /= len(validation_loader.dataset)
        val_metric /= len(validation_loader.dataset)

        if verbose:
            print("train loss: %.6f / train acc: %.2f / val loss: %.6f / val accuracy: %.2f / time: %.4f min" % (
                train_loss, 100 * train_metric, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
            print("-" * 10)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        # early_stopping(val_loss, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    model.load_state_dict(best_model_weights)

    return model, loss_history, metric_history


def show_history(num_epochs, loss_history, metric_history):
    plt.title("Train-Val Loss")

    plt.plot(range(1, num_epochs + 1), loss_history["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_history["val"], label="val")

    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")

    plt.legend()
    plt.show()

    plt.title('Train-Val Accuracy')
    plt.plot(range(1, num_epochs + 1), metric_history['train'], label='train')
    plt.plot(range(1, num_epochs + 1), metric_history['val'], label='val')

    plt.ylabel('Accuracy')
    plt.xlabel('Training Epochs')

    plt.legend()
    plt.show()
