import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
class EMNISTCNN(nn.Module):

    def __init__(self):
        super(EMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # logger.info("Looook here {}",x.size())
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
# old Neural Network
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class EMNISTCNN(nn.Module):

#     def __init__(self):
#         super(EMNISTCNN, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2))

#         self.fc = nn.Linear(7*7*32, 10)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = x.view(x.size(0), -1)

#         x = self.fc(x)

#         return x