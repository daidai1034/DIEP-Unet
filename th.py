import torch
from thop import profile
from models.UNet import UNet  # 导入你的模型

# 创建一个模型实例
model = UNet(1, 1).cuda()  # 这里假设你的模型在GPU上运行，如果在CPU上运行，使用 .cpu() 方法

# 创建一个示例输入张量（尺寸和数据类型需与模型匹配）
x = torch.randn(1, 1, 352, 352).cuda()  # 这里也是假设在GPU上运行

# 使用 thop 的 profile 函数计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(x,))
print(f"FLOPs: {flops / 1e9} GFLOPs")  # 将 FLOPs 转换为 GFLOPs（十亿次浮点运算）
print(f"Params: {params / 1e6} M")  # 将参数数量转换为百万

# 请根据你的实际模型和输入进行适当的修改
