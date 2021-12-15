## main.py文件
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse

# https://fyubang.com/2019/07/23/distributed-training3/

# [ddp]初始化 nccl 后端
torch.distributed.init_process_group(backend='nccl')

# [ddp]gpu id
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 构造模型
model = nn.Linear(10, 10).to(device)
model = DDP(model, device_ids=[local_rank],output_device=local_rank)

# 前向传播
outputs = model(torch.randn(20, 10).to(local_rank))
labels = torch.randn(20, 10).to(local_rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
# 后向传播
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()

