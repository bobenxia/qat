# qat

test qat in tensorrt:v8.2.1.8

 ## 使用
 主要脚本有四个：`model_train.py`，`model_eval.py`，`model_convert_to_onnx.py` 和 `model_calibrate_test.py`。分别介绍这四个脚本:

 ### 1、`model_train.py`
用于训练模型，包括：普通模型，量化后需要 finetune 的 Qat 模型。需要修改的配置文件在 `config/default_config.py` 中的 `_C.TRAIN` 部分
举例可以：
1. 使用 `checkpoint_normal.pth` 训练 `checkpoint_normal_train_best.pth`
2. 使用 `checkpoint_quant_calibrate.pth` 训练 `checkpoint_quant_train_best.pth`

 
### 2、`model_calibrate.py`
用于生成 qat 校准模型，校准后需要的 finetune 由上一个脚本实现。需要修改的配置文件在 `config/default_config.py` 中的 `_C.TEST` 和 `_C.QUANT`部分
举例：
1. 使用 `checkpoint_normal_train_best.pth` 校准生成 `checkpoint_quant_calibrate.pth`

### 3. `model_convert_to_onnx.py`
用于将 pth 模型转换成 onnx。需要修改的配置文件在 `config/default_config.py` 中的 `CONVERT_ONNX`部分
举例：
1. 使用 `checkpoint_quant_train_best.pth` 准换生成 `save_onnx/onnx_quant_True_dynamic_input_True.onnx`

额外说明：engine 生成没有脚本，直接调用 trtexec 完成。举例：
```
/usr/local/TensorRT-8.2.1.8/bin/trtexec --onnx=outputs/save_onnx/onnx_quant_True_dynamic_input_True.onnx --verbose --best --saveEngine=outputs/save_engine/engine_quant_True_dynamic_input_True.engine --minShapes=input:1x3x64x64 --maxShapes=input:256x3x224x224 --optShapes=input:64x3x224x224
```

### 4. `model_eval.py`
评测 pth、onnx 和 engine 模型的脚本。修改要评测模型的地址，位置在 `config/default_config.py` 中的 `EVAL`部分


## 记录数据
平台：rtx2060
idx|model |ACC@1|ACC@5|type| GPU latency speed(ms)
-|-|-|-|-|-
1|resnet50_origin|76.080|91.600|int8|0.6166
|||||fp16|0.8952
|||||fp32|2.7571
|||||best|0.6074
2|resbet50_quant_calibrate|75.420|91.430|int8|0.6034
3|resnet50_quant_QAT|75.730|91.650|int8|


 模型补充说明：

1. resnet50_orign:  构建的原始的 resnet50，
    1. 初始权重继承可以无，训练 60 epoch，初始 lr 0.01
    2. pth 模型为 `outputs/checkpoint_normal_train_best.pth`
2. resnet50_quant_calibrate: 搭建的量化的 resnet50，
    1. 初始权重继承 `checkpoint_normal_train_best.pth`，在此基础上进行校准
    2. pth 模型为 `outputs/checkpoint_quant_calibrate.pth`
3. resnet50_quant_QAT：搭建的量化的 resnet50，
    1. 初始权重继承校准后的权重 `outputs/checkpoint_quant_calibrate.pth`，在此基础上进行 finetune。
    2. pth 模型为 `outputs/checkpoint_quant_train_best.pth`


