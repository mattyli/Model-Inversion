import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

class VictimClassifier(nn.Module):
    def __init__(self, num_classes=500):
        super(VictimClassifier, self).__init__()
        
        # Layers 0 through 15: The Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),   # (0)
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),                           # (1)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (2)
            nn.ReLU(inplace=True),                                                  # (3)
            
            # Block 2
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # (4)
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),                           # (5)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (6)
            nn.ReLU(inplace=True),                                                  # (7)
            
            # Block 3
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # (8)
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1),                           # (9)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (10)
            nn.ReLU(inplace=True),                                                  # (11)
            
            # Block 4
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),# (12)
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1),                          # (13)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (14)
            nn.ReLU(inplace=True)                                                   # (15)
        )
        
        # Layers 16 through 18: The Classifier
        self.classifier = nn.Sequential(
            # Input features calculated: 1024 channels * 4 * 4 spatial size = 16384
            nn.Linear(1024, 512),                                                 # (16)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),                                                      # (17)
            nn.Linear(512, num_classes)                                            # (18)
        )

    def forward(self, x):
        # 1. Pass through convolutional layers
        x = self.features(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1) # Now shape is (Batch, 1024)
        
        # 3. Pass through fully connected layers
        x = self.classifier(x)
        
        return x