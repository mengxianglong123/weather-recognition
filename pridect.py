import torch
import torchvision.transforms as transforms
from PIL import Image
from config import Common
def pridect(imagePath, modelPath):
    '''
    预测函数
    :param imagePath: 图片路径
    :param modelPath: 模型路径
    :return:
    '''
    # 1. 读取图片
    image = Image.open(imagePath)
    # 2. 进行缩放
    image = image.resize(Common.imageSize)
    image.show()
    # 3. 加载模型
    model = torch.load(modelPath)
    model = model.to(Common.device)
    # 4. 转为tensor张量
    transform = transforms.ToTensor()
    x = transform(image)
    x = torch.unsqueeze(x, 0)  # 升维
    x = x.to(Common.device)
    # 5. 传入模型
    output = model(x)
    # 6. 使用argmax选出最有可能的结果
    output = torch.argmax(output)
    print("预测结果：",Common.labels[output.item()])

if __name__ == '__main__':
    pridect("D:/Download/2279624045.jpg","./model/weather-2022-10-14-07-36-57.pth")
