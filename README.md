# qat

test qat in tensorrt:v8.2.1.8

## 模型文件管理说明
 `outputs`路径下有保存的模型文件，使用 lfs 管理。

 拉取仓库的时候可以使用 `GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:bobenxia/qat.git 20211208_qat/` 先跳过大文件的下载。

 后续的拉取可以使用 `git lfs pull origin master`

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


more:
1. onnx engine evaluation 代码
2. nsight 查看 engine 内部结构。如果 ok，推广到其他模型。
3. https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity
4. https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins 调研一下 plugin 在 trt 中运行状态。
