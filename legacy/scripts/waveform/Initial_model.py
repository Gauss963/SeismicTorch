import torch
import torch.nn as nn
from torchviz import make_dot

class EarthquakeCNN(nn.Module):
    def __init__(self):
        super(EarthquakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 1), stride = (1, 1))
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 1), stride = (1, 1))
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 1), stride = (1, 1))
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(192, 16)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(16, 3)
        # self.softmax = nn.Softmax(dim = 1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.GELU3(x)
        x = self.maxpool2(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        # x = self.softmax(x)

        return x