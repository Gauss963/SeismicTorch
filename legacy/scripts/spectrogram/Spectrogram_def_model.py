import os
import sys
import torch
import torch.nn as nn

from io import StringIO
from thop import profile
from torchviz import make_dot

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = (3, 3), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(48)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 48, out_channels = 96, kernel_size = (3, 3), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(96)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = (3, 3), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(192)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(32640, 1280)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(1280, 64)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(64, 2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        x = self.fc3(x)

        return x

def do_check_SpectrogramCNN_model():
    # 保存原始的標準輸出流
    original_stdout = sys.stdout

    # 創建一個StringIO對象，用於捕捉打印內容
    output_buffer = StringIO()

    # 將標準輸出流設置為StringIO對象
    sys.stdout = output_buffer

    # 执行模型相關的打印操作
    model_plot = SpectrogramCNN()
    input_tensor = torch.randn(1, 3, 100, 150)
    output = model_plot(input_tensor)
    print(output.shape)

    total_params = sum([param.nelement() for param in model_plot.parameters()])
    print(f"Number of parameters: {total_params:.2f}")

    macs, params = profile(model_plot, inputs=(input_tensor,))
    flops = macs * 2
    print(f'The model has {flops} FLOPs')

    # 將標準輸出流還原為原始流
    sys.stdout = original_stdout

    # 獲取StringIO對象中的內容
    output_content = output_buffer.getvalue()

    # 將內容保存到文件
    with open('./time_evaluation_info/SpectrogramCNN_model_info.tex', 'w') as file:
        file.write('\\begin{frame}[fragile]{About Model: Structure}\n')
        file.write('\\tiny\n')
        file.write('{\n')
        file.write('\\begin{verbatim}\n')
        file.write(output_content)
        file.write('\\end{verbatim}\n')
        file.write('}\n')
        file.write('\end{frame}\n')

    # 生成模型結構圖並保存
    output = model_plot(input_tensor)
    dot = make_dot(output, params=dict(model_plot.named_parameters()), show_attrs=False)
    dot.render("./model_structure/SpectrogramCNN_Model", format = "pdf")
    os.remove("./model_structure/SpectrogramCNN_Model")


if __name__ == "__main__":
    do_check_SpectrogramCNN_model()