import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # resize
        self.re_conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.re_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.re_conv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # encoder
        self.enc_conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # decoder
        self.dec_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(128, 2, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # resize
        x = x.permute(0, 3, 2, 1)  # Swap dimensions 2 and 4
        re1 = self.re_conv1(x)
        re2 = self.re_conv2(re1)
        re3 = self.re_conv3(re2)

        # encoder
        x = re3.permute(0, 3, 2, 1)
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(enc1))
        enc3 = F.relu(self.enc_conv3(enc2))
        enc4 = F.relu(self.enc_conv4(enc3))
        
        # decoder
        dec1 = F.relu(self.dec_conv1(enc4))
        dec2 = F.relu(self.dec_conv2(torch.cat([dec1, enc3], dim=1)))
        dec3 = F.relu(self.dec_conv3(torch.cat([dec2, enc2], dim=1)))
        dec4 = self.dec_conv4(torch.cat([dec3, enc1], dim=1))
        
        return dec4
 