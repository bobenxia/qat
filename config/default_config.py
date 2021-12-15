from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.GPUS = (0,1,2,3,4,5,6,7)


# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK =True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.WEIGHTS = './outputs/checkpoint_normal.pth'
_C.MODEL.BN_MOMENTUM = 0.1


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "/data/xiazheng/tiny-imagenet-200"
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.VALID_SPLIT = 'val'
_C.DATASET.CROP_SIZE = 224
_C.DATASET.RESIZE_SIZE = 224
_C.DATASET.MEAN = (0.485, 0.456, 0.406)
_C.DATASET.STD = (0.229, 0.224, 0.225)
_C.DATASET.HFLIP_PROB = 0.5
_C.DATASET.AURO_AUGMENT_POLICY = 'imagenet'


# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.NUM_WORKERS = 8




# -----------------------------------------------------------------------------
# TRAIN 主要用于 model_train.py
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.QUANT =True
_C.TRAIN.WEIGHTS_NORMAL = "./outputs/checkpoint_normal.pth"
_C.TRAIN.WEIGHTS_QUANT = "./outputs/checkpoint_quant_calibrate.pth"
_C.TRAIN.AMP = False
_C.TRAIN.CLIP_GRAD_NORE = None
_C.TRAIN.EPOCH = 10
_C.TRAIN.SGD_LR = 0.001
_C.TRAIN.SGD_MOMENTUM = 0.9
_C.TRAIN.BEST_NORMAL_SAVE = "./outputs/checkpoint_normal_train_best.pth"
_C.TRAIN.BEST_QUANT_SAVE = "./outputs/checkpoint_quant_train_best.pth"




# -----------------------------------------------------------------------------
# TEST 和 QUANTIZATION 主要用于 model_calibrate_test.py
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()

_C.TEST.GPUS = (0,)
_C.TEST.QUANT = True
_C.TEST.COMPARE = True
_C.TEST.WEIGHTS_NORMAL = "./outputs/checkpoint_normal_train_best.pth"
_C.TEST.WEIGHTS_QUANT = "./outputs/checkpoint_quant_calibrate.pth"

# -----------------------------------------------------------------------------
# QUANTIZATION
# -----------------------------------------------------------------------------
_C.QUANT = CN()
_C.QUANT.CALIB = False
_C.QUANT.CALIB_METHOD = 'max' # ('max', 'histogram')
_C.QUANT.NUM_CALIBED_BATCH = 30
_C.QUANT.HISTOGRAM_METHOD = 'mse'
_C.QUANT.CALIB_SAVE = './outputs/checkpoint_quant_calibrate.pth'




# -----------------------------------------------------------------------------
# CONVERT_ONNX 主要用于 model_convert_to_onnx.py
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CONVERT_ONNX
# -----------------------------------------------------------------------------
_C.CONVERT_ONNX = CN()

_C.CONVERT_ONNX.QUANT = False
_C.CONVERT_ONNX.QUANT_WEIGHTS = './outputs/checkpoint_quant_calibrate.pth'
_C.CONVERT_ONNX.NORMAL_WEIGHTS = './outputs/checkpoint_normal_train_best.pth'
_C.CONVERT_ONNX.QUANT_SAVE = './outputs/onnx_quant.onnx'
_C.CONVERT_ONNX.NORMAL_SAVE = './outputs/onnx_normal.onnx'
_C.CONVERT_ONNX.DYNAMIC = True
_C.CONVERT_ONNX.ONNX_FILE = ''
_C.CONVERT_ONNX.ONNX_FILE_WITH_DYNAMIC = ''
_C.CONVERT_ONNX.OPSET_VERSION = 13
_C.CONVERT_ONNX.VERBOSE = False
_C.CONVERT_ONNX.TRAING = False
_C.CONVERT_ONNX.ENABLE_ONNX_CHECKER = False
_C.CONVERT_ONNX.DO_CONSTANT_FOLDING = True
_C.CONVERT_ONNX.INPUT_NAMES = ['input']
_C.CONVERT_ONNX.OUTPUT_NAMES = ['output']




# -----------------------------------------------------------------------------
# CONVERT_ONNX 主要用于 model_convert_to_onnx.py
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CONVERT_ONNX
# -----------------------------------------------------------------------------
_C.EVAL = CN()

_C.EVAL.QUANT = False
_C.EVAL.QUANT_WEIGHTS = './outputs/checkpoint_quant_train_best.pth'
_C.EVAL.NORMAL_WEIGHTS = './outputs/checkpoint_normal_train_best.pth'
_C.EVAL.ONNX_WEIGHTS = './outputs/onnx_normal.onnx'