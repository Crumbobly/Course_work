import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))  # пулинг только по ширине
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(1,2), stride=(1,2))
        self.upconv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=(1,2), stride=(1,2))
        self.upconv1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)                  # [B, 32, 15, 1000]
        d2 = self.down2(self.pool1(d1))    # [B, 64, 15, 500]
        b = self.bottleneck(self.pool2(d2))# [B, 128, 15, 250]

        u2 = self.up2(b)                   # [B, 64, 15, 500]
        u2 = self.upconv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(u2)                  # [B, 32, 15, 1000]
        u1 = self.upconv1(torch.cat([u1, d1], dim=1))

        return self.final(u1)  # для CrossEntropyLoss НЕ нужно sigmoid
