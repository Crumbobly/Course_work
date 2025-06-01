import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlusModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()

        # Encoder
        self.conv00 = ConvBlock(in_channels, 32)
        self.pool0 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv10 = ConvBlock(32, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv20 = ConvBlock(64, 128)

        # Decoder с dense skip connections
        self.up01 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv01 = ConvBlock(32 + 32, 32)

        self.up11 = nn.ConvTranspose2d(128, 64, kernel_size=(1, 2), stride=(1, 2))
        self.conv11 = ConvBlock(64 + 64, 64)

        self.up02 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv02 = ConvBlock(32 + 32 + 32, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))

        x01 = self.conv01(torch.cat([x00, self.up01(x10)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up11(x20)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up02(x11)], dim=1))

        return self.final(x02)
