import torch.nn as nn
from einops import rearrange

# A modular block combining Convolution, Batch Normalization, and Activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool: 
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# ResNet-9: Optimized for fast, high-accuracy training on CIFAR-10
class SimpleNet(nn.Module): # Kept the same class name so train.py doesn't break
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        
        # Initial preparation block
        self.prep = ConvBlock(in_channels, 64)
        
        # Layer 1 with a Residual Skip Connection
        self.layer1_head = ConvBlock(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(
            ConvBlock(128, 128), 
            ConvBlock(128, 128)
        )
        
        # Layer 2 (Transition)
        self.layer2 = ConvBlock(128, 256, pool=True)
        
        # Layer 3 with a Residual Skip Connection
        self.layer3_head = ConvBlock(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(
            ConvBlock(512, 512), 
            ConvBlock(512, 512)
        )
        
        # Final Classifier
        self.pool = nn.MaxPool2d(4)
        self.dropout = nn.Dropout(0.2) # Prevents the deeper network from overfitting
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Pass through Prep
        out = self.prep(x)
        
        # Pass through Layer 1 (Head + Residual)
        out = self.layer1_head(out)
        out = self.layer1_residual(out) + out  # <--- This is the famous Skip Connection!
        
        # Pass through Layer 2
        out = self.layer2(out)
        
        # Pass through Layer 3 (Head + Residual)
        out = self.layer3_head(out)
        out = self.layer3_residual(out) + out  # <--- Another Skip Connection!
        
        # Pool, Flatten, and Classify
        out = self.pool(out)
        out = rearrange(out, 'b c h w -> b (c h w)') # Using einops to flatten
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out
