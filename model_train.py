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



    


    # torch.manual_seed(2)
    # data_input = torch.rand([1,3,1024,1920])
    # model.eval()
    # out = model(data_input)

    # if use_dynamic:
    #     torch.onnx.export(model, data_input, onnx_file, 
    #                     opset_version=13, 
    #                     verbose=False, 
    #                     training=False,
    #                     enable_onnx_checker=False,
    #                     do_constant_folding=True,
    #                     input_names = ['input'],
    #                     output_names = ['output'],
    #                     dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3: 'width'},
    #                                 'output' : {0 : 'batch_size', 2: 'height', 3: 'width'}})
    # else:
    #     torch.onnx.export(model, data_input, onnx_file, 
    #                     opset_version=13, 
    #                     verbose=False, 
    #                     training=False,
    #                     enable_onnx_checker=False,
    #                     do_constant_folding=True,
    #                     input_names = ['input'],
    #                     output_names = ['output'])
    # print(f"onnx convert success in {onnx_file}")

    # import onnx
    # import onnxsim
    # model_onnx = onnx.load(onnx_file)
    # # onnx.checker.check_model(model)
    # if use_dynamic:
    #     model_opt, check_ok = onnxsim.simplify(model_onnx,dynamic_input_shape=True, input_shapes={'input':[1, 3, 1024, 1920]})
    # else:
    #     model_opt, check_ok = onnxsim.simplify(model_onnx,dynamic_input_shape=False, input_shapes={'input':[1, 3, 1024, 1920]})
    # onnx.save(model_opt, onnx_sim_file)
    # print(f"onnx simplify success in {onnx_sim_file}")

    # import onnxruntime as rt
    # sess = rt.InferenceSession(onnx_file)
    # out_onnx = sess.run(None, {sess.get_inputs()[0].name:data_input.cpu().detach().numpy()})
    # sess2 = rt.InferenceSession(onnx_sim_file)
    # out_onnx2 = sess2.run(None, {sess2.get_inputs()[0].name:data_input.cpu().detach().numpy()})

    # # 结果对比
    # out = out.cpu().detach().numpy()
    # out_onnx = out_onnx[0]
    # out_onnx2 = out_onnx2[0]

    # print(out.sum(), out_onnx.sum(), out_onnx2.sum())
    # print(out.shape, out_onnx.shape, out_onnx2.shape)
    # if not np.allclose(out, out_onnx, rtol=1e-1, atol=1e-2):
    #         print(
    #             'The outputs are different between Pytorch and ONNX')
    # if not np.allclose(out_onnx, out_onnx2, rtol=1e-1, atol=1e-2):
    #         print(
    #             'The outputs are different between ONNX and ONNX_sim')

if __name__=='__main__':
    main()
