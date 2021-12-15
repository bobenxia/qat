import os

import torch
from .resnet_quant import resnet50


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