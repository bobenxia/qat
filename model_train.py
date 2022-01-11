import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn

from config.default_config import _C as config
from dataset.build_dataloader import build_train_loader, build_valid_loader
from dataset.build_dataset import (build_train_dataset_and_sampler,
                                   build_valid_dataset_and_sampler)
from train_and_eval.evaluate import evaluate
from train_and_eval.train import train_one_epoch
from models.get_model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train network")

    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    return args
    

def main():
    args = parse_args()

    # 0- cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device(f'cuda:{args.local_rank}')

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    
    # 1- load model
    if config.TRAIN.QUANT:
        quant_nn.TensorQuantizer.use_fb_fake_quant = True

        print("Load the model as a quantized version")
        quant_desc_input = QuantDescriptor(calib_method=config.QUANT.CALIB_METHOD)  
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantAdaptiveAvgPool2d.set_default_quant_desc_input(quant_desc_input)

        quant_modules.initialize()
        model = get_model(config.TRAIN.QUANT, config.TRAIN.WEIGHTS_QUANT)
        save_path = config.TRAIN.BEST_QUANT_SAVE
        quant_modules.deactivate()
    else:
        print("Load the model as a normal version")
        model = get_model(config.TRAIN.QUANT, config.TRAIN.WEIGHTS_NORMAL)
        save_path = config.TRAIN.BEST_NORMAL_SAVE
    
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model.to(device)


    # 2- data loader
    dataset_train, train_sampler= build_train_dataset_and_sampler(config, distributed)
    dataset_valid, valid_sampler= build_valid_dataset_and_sampler(config, distributed)
    data_loader_train = build_train_loader(dataset_train, train_sampler, config, distributed)
    data_loader_test = build_valid_loader(dataset_valid, valid_sampler, config, distributed)


    # 3- train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = config.TRAIN.SGD_LR, momentum=config.TRAIN.SGD_MOMENTUM)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_acc1 = 0

    for epoch in range(config.TRAIN.EPOCH):
        if distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch, config=config)
        lr_scheduler.step()
        acc1 = evaluate(model, criterion, data_loader_test, device)

        model_save = model.module if distributed else model

        if best_acc1 < acc1:
            best_acc1 = acc1
            print("get the new highest score: ", acc1)
            torch.save(model_save.state_dict(), save_path)

    print("Finally, the highest score: ", best_acc1)

if __name__=='__main__':
    main()
