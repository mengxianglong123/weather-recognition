import time

import torch
# 项目配置文件

class Common:
    '''
    通用配置
    '''
    basePath = "D:/Data/weather/source/all/"  # 图片文件基本路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 设备配置
    imageSize = (224,224) # 图片大小
    labels = ["cloudy","haze","rainy","shine","snow","sunny","sunrise","thunder"] # 标签名称/文件夹名称


class Train:
    '''
    训练相关配置
    '''
    batch_size = 128
    num_workers = 0  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    lr = 0.001
    epochs = 40
    logDir = "./log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置
    modelDir = "./model/" # 模型存放位置



