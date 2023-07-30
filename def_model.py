import torch
import torch.nn as nn
from torchviz import make_dot
from thop import profile

class EarthquakeCNN(nn.Module):
    def __init__(self):
        super(EarthquakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 1), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 1), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 1), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(384, 64)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(64, 16)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(16, 3)
        

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

if __name__ == "__main__":
    model_plot = EarthquakeCNN()
    input_tensor = torch.randn(1, 1, 80, 3)
    output = model_plot(input_tensor)
    print(output)
    print(output.shape)

    total = sum([param.nelement() for param in model_plot.parameters()])
    print("Number of parameter: %.2f" % (total))


    macs, params = profile(model_plot, inputs = (input_tensor, ))
    flops = macs * 2
    print(f'The model has {flops} FLOPs')