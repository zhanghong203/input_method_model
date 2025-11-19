import time

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer


def train_per_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 设备
    :return: 当前epoch的平均损失
    """
    # 设置训练模式
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="训练"):
        # input.shape (batch_size, seq_len) target.shape(batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        # 前向传播
        outputs = model(inputs)
        # output.shape (batch_size, vocab_size)
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 累加每一轮的损失
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 加载数据集
    dataloader = get_dataloader()
    # 3. 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODEL_DIR / 'vocab.txt')
    # 4. 创建模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. 准备tensorboard writer
    best_loss = float('inf')
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S"))
    # 开始训练
    for epoch in range(config.EPOCHS):
        print("=" * 30, f"Epoch: {epoch + 1}")
        # 训练
        loss = train_per_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"\ntrain loss: {loss}")
        # TensorBoard可视化loss
        writer.add_scalar('loss', loss, epoch)
        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODEL_DIR / 'best.pt')
            print("模型保存成功")


if __name__ == '__main__':
    train()
