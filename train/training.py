import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入你的代码模块
from model.zzyNet import zzyNet
from datasets.datasetsLoader import KITTIDataset
from Loss.ReprojectionLoss import ReprojectionLoss
from Loss.SmoothnessLoss import SmoothnessLoss
from utils.optimizers import get_step_schedule_with_warmup

# 配置训练参数
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = KITTIDataset(depth_type=None, single_intrinsics=False, path="2011_09_26/2011_09_26_drive_0001_sync")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 初始化模型
model = zzyNet(cfg={})  # 需要正确的 config 参数
model.to(DEVICE)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = get_step_schedule_with_warmup(optimizer, lr_start=0.1, warmup_epochs=5, epoch_size=len(train_loader), step_size=10, gamma=0.5)

# 定义损失函数
reproj_loss = ReprojectionLoss()
smooth_loss = SmoothnessLoss()

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        # 获取输入数据
        batch = {k: v.to(DEVICE) for k, v in batch.items()}  # 将 batch 数据转移到 GPU
        output = model(batch)

        # 计算损失
        loss_depth = reproj_loss(output['predictions'], batch)
        loss_smooth = smooth_loss(output['predictions'], batch)
        loss = loss_depth + 0.1 * loss_smooth  # 平衡两个损失

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()  # 更新学习率
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), "trained_model.pth")
print("模型训练完成并已保存！")
