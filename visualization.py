import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
from models.DIEPUnet import*

# 加载模型
model = UNet()
model.load_state_dict(torch.load('hook/MYnet.pth'))

# 数据
input_image = Image.open('hook/10_150_img.png')  # PIL
mask_image = Image.open('hook/10_150_mask.png')
transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
input_image = transform(input_image).unsqueeze(0)  # [1, 1, 352, 352]
mask_image = transform(mask_image).unsqueeze(0)

# 定义钩子函数，获取指定层名称的特征
activation = {}  # 保存获取的输出
gradient = None

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_gradient(module, grad_input, grad_output):
    global gradient
    gradient = grad_output[0]
# 注册钩子
model.eval()

model.hgca_3.register_forward_hook(get_activation('5'))  # 为layer1中第2个模块的bn3注册钩子
model.hgca_3.register_backward_hook(get_gradient)  # 注册梯度钩子



# 前向传播
output,_,_,_,_ = model(input_image)
output = torch.sigmoid(output)
# 取出对应特征图
bn3 = activation['5']  # bn3=torch.Size([1, 32, 176, 176])
print('bn3:',bn3.shape)
print('bn3:',type(bn3))
# 计算损失和反向传播获取梯度
loss = torch.mean(output)#执行前向传播得到输出，计算损失并执行反向传播，收集特定层的输出值以及计算了梯度值
loss.backward()

# 计算CAM
weights = torch.mean(gradient, axis=(2, 3), keepdim=True)
cam = torch.sum(weights * bn3, axis=1, keepdim=True)#---------------权重x提取层
cam = F.relu(cam)  # ReLU函数


# 将CAM转换为伪彩色热力图
cam = cam - cam.min()
cam = cam / cam.max()
cam = cam.cpu().numpy()  # 转为numpy数组
cam = np.uint8(255 * cam)  # 转为0-255范围的整数
cam = cv2.applyColorMap(cam[0, 0], cv2.COLORMAP_JET)  # 应用伪彩色映射
print('cam:',cam.shape)
print('cam:',type(cam))

cam = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_NEAREST)
# 保存cam为图像文件
cv2.imwrite('hook/test_img/cam.png', cam)


# start=0
# end=15
# for i in range(start, end + 1):
#     bn_slice = bn3[0, i, :, :].cpu().numpy()
#     # 确保 bn_slice 是 uint8 类型
#     bn_slice = np.uint8(bn_slice)

#     # 转换为单通道灰度图像（CV_8UC1）
#     bn_slice_gray = cv2.cvtColor(bn_slice, cv2.COLOR_GRAY2BGR)

#     # 应用伪彩色映射
#     colored_slice = cv2.applyColorMap(bn_slice_gray, cv2.COLORMAP_JET)

#     # 保存为图像文件
#     cv2.imwrite(f"hook/test_img/bn{i}.png", colored_slice)

