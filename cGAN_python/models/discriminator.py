import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # encoder
        self.dec_conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 8, kernel_size=4, stride=2, padding=1)
        self.dec_conv4 = nn.Conv2d(8, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # encoder
        dec1 = F.relu(self.dec_conv1(x))
        dec2 = F.relu(self.dec_conv2(dec1))
        dec3 = F.relu(self.dec_conv3(dec2))
        dec4 = torch.sigmoid(self.dec_conv4(dec3))
        
        return dec4