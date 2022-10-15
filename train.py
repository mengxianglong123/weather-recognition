# 训练部分
import time
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from config import Common, Train
from model import model as weatherModel
from data_loader import trainLoader, valLoader
from torch import optim

# 1. 获取模型
model = weatherModel
model.to(Common.device)
# 2. 定义损失函数
criterion = nn.CrossEntropyLoss()
# 3. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 创建writer
writer = SummaryWriter(log_dir=Train.logDir, flush_secs=500)


def train(epoch):
    '''
    训练函数
    '''
    # 1. 获取dataLoader
    loader = trainLoader
    # 2. 调整为训练状态
    model.train()
    print()
    print('========== Train Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    for data, label in loader:
        data, label = data.to(Common.device), label.to(Common.device)  # 加载到对应设备
        batchAcc = 0  # 单批次正确率
        batchCorrectNum = 0  # 单批次正确个数
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 获取模型输出
        loss = criterion(output, label)  # 计算损失
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 更新参数
        epochLoss += loss.item() * data.size(0)  # 计算损失之和
        # 计算正确预测的个数
        labels = torch.argmax(label, dim=1)
        outputs = torch.argmax(output, dim=1)
        for i in range(0, len(labels)):
            if labels[i] == outputs[i]:
                correctNum += 1
                batchCorrectNum += 1
        batchAcc = batchCorrectNum / data.size(0)
        print("Epoch:{}\t TrainBatchAcc:{}".format(epoch, batchAcc))

    epochLoss = epochLoss / len(trainLoader.dataset)  # 平均损失
    epochAcc = correctNum / len(trainLoader.dataset)  # 正确率
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    writer.add_scalar("train_loss", epochLoss, epoch)  # 写入日志
    writer.add_scalar("train_acc", epochAcc, epoch)  # 写入日志
    return epochAcc

def val(epoch):
    '''
    验证函数
    :param epoch: 轮次
    :return:
    '''
    # 1. 获取dataLoader
    loader = valLoader
    # 2. 初始化损失、准确率列表
    valLoss = []
    valAcc = []
    # 3. 调整为验证状态
    model.eval()
    print()
    print('========== Val Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(Common.device), label.to(Common.device)  # 加载到对应设备
            batchAcc = 0  # 单批次正确率
            batchCorrectNum = 0  # 单批次正确个数
            output = model(data)  # 获取模型输出
            loss = criterion(output, label)  # 计算损失
            epochLoss += loss.item() * data.size(0)  # 计算损失之和
            # 计算正确预测的个数
            labels = torch.argmax(label, dim=1)
            outputs = torch.argmax(output, dim=1)
            for i in range(0, len(labels)):
                if labels[i] == outputs[i]:
                    correctNum += 1
                    batchCorrectNum += 1
            batchAcc = batchCorrectNum / data.size(0)
            print("Epoch:{}\t ValBatchAcc:{}".format(epoch, batchAcc))

        epochLoss = epochLoss / len(valLoader.dataset)  # 平均损失
        epochAcc = correctNum / len(valLoader.dataset)  # 正确率
        print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
        writer.add_scalar("val_loss", epochLoss, epoch)  # 写入日志
        writer.add_scalar("val_acc", epochAcc, epoch)  # 写入日志
    return epochAcc

if __name__ == '__main__':
    maxAcc = 0.75
    for epoch in range(1,Train.epochs + 1):
        trainAcc = train(epoch)
        valAcc = val(epoch)
        if valAcc > maxAcc:
            maxAcc = valAcc
            # 保存最大模型
            torch.save(model, Train.modelDir + "weather-" + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime()) + ".pth")
    # 保存模型
    torch.save(model,Train.modelDir+"weather-"+time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime())+".pth")


