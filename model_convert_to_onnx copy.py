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
    gpus = list(config.TEST.GPUS)
    distributed = len(gpus) > 1
    if distributed:
        raise ValueError('Convert only supports single core.')
    device = torch.device(f'cuda:{args.local_rank}')

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    
    # 1- load model
    if config.CONVERT_ONNX.QUANT:
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
        model = get_model(config.CONVERT_ONNX.QUANT, config.CONVERT_ONNX.QUANT_WEIGHTS)
        quant_modules.deactivate()
    else:
        print("Load the model as a normal version")
        model = get_model(config.CONVERT_ONNX.QUANT, config.CONVERT_ONNX.NORMAL_WEIGHTS)
    
    model.to(device)


    # 2- Convert onnx 
    torch.manual_seed(2)
    data_input = torch.rand([1,3,224,224]).to(device)

    onnx_file = config.CONVERT_ONNX.QUANT_SAVE \
            if config.CONVERT_ONNX.QUANT else config.CONVERT_ONNX.NORMAL_SAVE
    opset_version = config.CONVERT_ONNX.OPSET_VERSION
    verbose = config.CONVERT_ONNX.VERBOSE
    training = config.CONVERT_ONNX.TRAING
    enable_onnx_checker = config.CONVERT_ONNX.ENABLE_ONNX_CHECKER
    do_constant_folding = config.CONVERT_ONNX.DO_CONSTANT_FOLDING
    input_names = config.CONVERT_ONNX.INPUT_NAMES
    output_names = config.CONVERT_ONNX.OUTPUT_NAMES
    dynamic_axes = {'input' : {0 : 'batch_size', 2: 'height', 3: 'width'}, 
                    'output' : {0 : 'batch_size', 2: 'height', 3: 'width'}} \
            if config.CONVERT_ONNX.DYNAMIC else None

    torch.onnx.export(model, data_input, onnx_file, 
            opset_version=opset_version, 
            verbose=verbose, 
            training=training,
            enable_onnx_checker=enable_onnx_checker,
            do_constant_folding=do_constant_folding,
            input_names = input_names,
            output_names = output_names,
            dynamic_axes=dynamic_axes)
    print(f"onnx convert success in {onnx_file}")

    # 2-1 onnx simplify
    model_onnx = onnx.load(onnx_file)   
    # # onnx.checker.check_model(model)
    if config.CONVERT_ONNX.DYNAMIC:
        model_opt, check_ok = onnxsim.simplify(model_onnx,dynamic_input_shape=True, input_shapes={'input':[1, 3, 224, 224]})
    else:
        model_opt, check_ok = onnxsim.simplify(model_onnx,dynamic_input_shape=False, input_shapes={'input':[1, 3, 224, 224]})
    onnx.save(model_opt, onnx_file)
    print(f"onnx simplify success in {onnx_file}")


    # 3- sync test
    model.eval()
    out = model(data_input)
    sess = rt.InferenceSession(onnx_file)
    out_onnx = sess.run(None, {sess.get_inputs()[0].name:data_input.cpu().detach().numpy()})

    out = out.cpu().detach().numpy()
    out_onnx = out_onnx[0]

    print(out.sum(), out_onnx.sum())
    print(out.shape, out_onnx.shape)
    if not np.allclose(out, out_onnx, rtol=1e-1, atol=1e-2):
            print(
                'The outputs are different between Pytorch and ONNX')


if __name__=='__main__':
    main()
