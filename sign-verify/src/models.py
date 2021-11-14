import torch as T
import torch.nn as nn
from torchvision import models


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 8),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_features = 128 * 21 * 9
        self.fc = nn.Sequential(
            nn.Linear(self.conv_features, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.conv_features)
        return self.fc(x)


class ResConvNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResConvNet, self).__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=7, padding=3)
        )
        self.resnet = models.resnet18(pretrained=pretrained)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ft = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ft, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.start_conv(x)
        x = self.resnet(x)
        return x


class VggConvNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VggConvNet, self).__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=7, padding=3)
        )
        self.vgg = models.vgg19_bn(pretrained=pretrained)
        num_ft = 1000
        self.fc = nn.Sequential(
            nn.Linear(num_ft, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.start_conv(x)
        x = self.vgg(x)
        x = self.fc(x)
        return x




