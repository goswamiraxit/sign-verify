import torch as T
import torch.nn as nn

from models import ResConvNet

model = nn.Sequential(
            nn.Conv2d(2, 16, 8),
            nn.MaxPool2d(5),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2)
)

# model = ResConvNet()

x = T.rand(1, 2, 1375, 674)
y = model(x)

print(x.size())
print(y.size())


