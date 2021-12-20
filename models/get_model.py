import os

import torch

from .trt_engine import TRTModel
from .resnet_quant import resnet50
from .onnx_model import ONNXModel


def get_model(use_quantize=False, weights_path=''):
    # if not use_quantize:
    #     model = torchvision.models.resnet50(pretrained=False, progress=False)
    # else:
    #     model = torchvision.models.resnet50(pretrained=False, progress=False)
    model = resnet50(pretrained=False, progress=False, quantized=use_quantize)

    if not os.path.exists(weights_path):
        print("Pretrain weight:{} doesn't exists".format(weights_path))
    else:
        print("loading weights:{}".format(weights_path) )
        pretrained_state = torch.load(weights_path, map_location='cuda:0')
        model.load_state_dict(pretrained_state)        

    return model


def get_model_onnx(weights_path):
    if not os.path.exists(weights_path):
        raise Exception("Pretrain weight:{} doesn't exists".format(weights_path))
    model = ONNXModel(weights_path)

    return model

def get_model_engine(weights_path, max_batch_size, input_shape):
    if not os.path.exists(weights_path):
        raise Exception("Pretrain weight:{} doesn't exists".format(weights_path))
    model = TRTModel(weights_path, max_batch_size, input_shape)

    return model