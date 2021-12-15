import collections
from pytorch_quantization import calib
from tqdm import tqdm
import torch
import pytorch_quantization.nn as quant_nn


def calibrate_model(model, data_loader, num_calib_batches, calib_method, histogram_method, save_path):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: model
            data_loader: calibration data set
            num_calib_batches: amount of calibration state files
            calib_method: type of calibration to use (max/histogram)
            histogram_method: type of method if use histogram (mse/entropy)
    """
    if num_calib_batches > 0:
        print("Calibrating model")
        with torch.inference_mode():
            collect_stats(model, data_loader, num_calib_batches)

    if calibrate_model == 'max':
        compute_amax(model, method=calib_method)
    else:
        compute_amax(model, method=histogram_method)

    torch.save(model.state_dict(), save_path)

        

def collect_stats(model, data_loader, num_batches):
    "Feed data to the network and collect statistic"

    # Enable Calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    
    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable Calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    model.cuda()
    