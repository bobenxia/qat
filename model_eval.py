import argparse
import collections
import datetime
import logging
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime as rt
import onnxsim
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn
from torch._C import device
from torch.distributed.distributed_c10d import Backend
from torch.types import Number
from torch.utils.data import dataset
from tqdm import tqdm

from calibrate.calibrate import calibrate_model
from config.default_config import _C as config
from dataset.build_dataloader import build_train_loader, build_valid_loader
from dataset.build_dataset import (build_train_dataset_and_sampler,
                                   build_valid_dataset_and_sampler)
from train_and_eval.evaluate import evaluate, evaluate_onnx
from train_and_eval.train import train_one_epoch
from models.get_model import get_model, get_model_onnx
from utils.accuracy import to_numpy


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
    gpus = list(config.TEST.GPUS)
    distributed = len(gpus) > 1
    if distributed:
        raise ValueError('Convert only supports single core.')
    device = torch.device(f'cuda:{args.local_rank}')

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    
    # 1- load model
    if config.EVAL.QUANT:
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
        model = get_model(config.EVAL.QUANT, config.EVAL.QUANT_WEIGHTS)
        quant_modules.deactivate()
    else:
        print("Load the model as a normal version")
        model = get_model(config.EVAL.QUANT, config.EVAL.NORMAL_WEIGHTS)
    
    model_onnx = get_model_onnx(config.EVAL.ONNX_WEIGHTS)
    model.to(device)


    # 2- data
    dataset_train, train_sampler= build_train_dataset_and_sampler(config, distributed)
    dataset_valid, valid_sampler= build_valid_dataset_and_sampler(config, distributed)
    data_loader_train = build_train_loader(dataset_train, train_sampler, config, distributed)
    data_loader_test = build_valid_loader(dataset_valid, valid_sampler, config, distributed)
    criterion = nn.CrossEntropyLoss()

    # 3- eval
    print("Onnx model eval:")
    evaluate_onnx(model_onnx, criterion, data_loader_test, device)
    print("Torch model eval:")
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device)


if __name__=='__main__':
    main()
