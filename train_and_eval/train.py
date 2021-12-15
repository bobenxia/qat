import time
import torch
import torch.nn as nn

import utils


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader,20, header)):
        start_time = time.time()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        scaler = torch.cuda.amp.GradScaler() if config.TRAIN.AMP else None
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD_NORE is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORE)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.TRAIN.CLIP_GRAD_NORE is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD_NORE)
            optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1,5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        