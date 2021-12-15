## main.py文件
import torch
import torch.nn as nn
import torch.optim as optim

# 构造模型
model = nn.Linear(10, 10).to('cuda')

for i in range(10000):
    # print(i)
    # 前向传播
    outputs = model(torch.randn(20, 10).to('cuda'))
    labels = torch.randn(20, 10).to('cuda')
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    # 后向传播
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()

