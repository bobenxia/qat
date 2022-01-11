import argparse
import os

import torch
import torch.utils.data
import torchvision
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from torch import nn
from torch.backends import cudnn

from calibrate.calibrate import calibrate_model
from config.default_config import _C as config
from dataset.build_dataloader import build_train_loader, build_valid_loader
from dataset.build_dataset import (build_train_dataset_and_sampler,
                                   build_valid_dataset_and_sampler)
from train_and_eval.evaluate import evaluate
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
    gpus = list(config.TEST.GPUS)
    distributed = len(gpus) > 1
    if distributed:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))


    # 1- load model
    if config.TEST.QUANT:
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
        model  = get_model(config.TEST.QUANT, config.TEST.WEIGHTS_QUANT)
        quant_modules.deactivate()         
    else:
        print("Load the model as a normal version")
        model  = get_model(config.TEST.QUANT, config.TEST.WEIGHTS_NORMAL)
    model.to(device)


    # 2- data loader
    dataset_train, train_sampler= build_train_dataset_and_sampler(config, distributed)
    dataset_valid, valid_sampler= build_valid_dataset_and_sampler(config, distributed)
    data_loader_train = build_train_loader(dataset_train, train_sampler, config, distributed)
    data_loader_test = build_valid_loader(dataset_valid, valid_sampler, config, distributed)
    criterion = nn.CrossEntropyLoss()


    # 3- calibrate model
    if config.TEST.QUANT and config.QUANT.CALIB:
        num_calib_batches=config.QUANT.NUM_CALIBED_BATCH
        calib_method=config.QUANT.CALIB_METHOD
        histogram_method = config.QUANT.HISTOGRAM_METHOD
        save_method = config.QUANT.CALIB_SAVE
        calibrate_model(model, data_loader_train, num_calib_batches, calib_method, histogram_method, save_method)
    

    # 4- evaluate the calibrated model
    with torch.no_grad():
        evaluate(model, criterion, data_loader_test, device)
        if config.TEST.COMPARE:
            print("compared model eval:")
            model_compare  = get_model(config.TEST.QUANT, config.TEST.WEIGHTS_NORMAL)
            evaluate(model_compare.to(device), criterion, data_loader_test, device)


if __name__=='__main__':
    main()
