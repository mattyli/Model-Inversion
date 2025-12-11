import torch
import torch.nn as nn
import torch.nn.functional as F

class VictimClassifier(nn.Module):
    def __init__(self, num_classes=500):
        super(VictimClassifier, self).__init__()
        
        # Layers 0 through 15: The Feature Extractor
        # CHANGE: Reduced width by 50% (e.g., 128->64, 1024->512)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),    # (0)
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),                            # (1)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (2)
            nn.ReLU(inplace=True),                                                  # (3)
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # (4)
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1),                           # (5)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (6)
            nn.ReLU(inplace=True),                                                  # (7)
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # (8)
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1),                           # (9)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (10)
            nn.ReLU(inplace=True),                                                  # (11)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), # (12)
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1),                           # (13)
            nn.MaxPool2d(kernel_size=2, stride=2),                                  # (14)
            nn.ReLU(inplace=True)                                                   # (15)
        )
        
        # Layers 16 through 18: The Classifier
        # CHANGE: Reduced input from 1024 to 512 (matching last conv layer)
        # CHANGE: Bottlenecked hidden layer to 512 instead of expanding to 2650
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),                                                    # (16)
            nn.ReLU(inplace=True),                                                  # Added ReLU
            nn.Dropout(p=0.5),                                                      # (17)
            nn.Linear(512, num_classes)                                             # (18)
        )

    def forward(self, x):
        x = self.features(x)
        
        # Global Average Pooling reduces (Batch, 512, 4, 4) -> (Batch, 512, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1) 
        
        x = self.classifier(x)
        
        return x