import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False),
            nn.Hardsigmoid()
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        channel_att = self.channel_gate(x)
        x = x * channel_att
        spatial_att = torch.cat([torch.mean(x, dim=1, keepdim=True), 
                                 torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        spatial_att = self.spatial_gate(spatial_att)
        spatial_att = torch.sigmoid(spatial_att)
        x = x * spatial_att
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(ConvBlock, self).__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                            padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs = nn.Hardswish()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.hs(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.cbam(x)
        return x

class ProposedMynet(nn.Module):
    def __init__(self):
        super(ProposedMynet, self).__init__()
        self.input = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=False)
        self.bn_input = nn.BatchNorm2d(16)
        self.hs_input = nn.Hardswish()
        self.block1 = ConvBlock(16, 48, stride=2)
        self.block2 = ConvBlock(48, 96, stride=2)
        self.block3 = ConvBlock(96, 48, stride=1)
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.fuse1 = ConvBlock(96, 48, stride=1)
        self.up2 = nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1)
        self.fuse2 = ConvBlock(48, 32, stride=1)
        self.output = nn.Conv2d(32, 3, kernel_size=1, stride=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x0 = self.input(x)
        x0 = self.bn_input(x0)
        x0 = self.hs_input(x0)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x_up1 = self.up1(x2)
        x_fuse1 = torch.cat([x_up1, x1], dim=1)
        x_fuse1 = self.fuse1(x_fuse1)
        x_up2 = self.up2(x_fuse1)
        x_fuse2 = torch.cat([x_up2, x0], dim=1)
        x_fuse2 = self.fuse2(x_fuse2)
        out = self.output(x_fuse2)
        out = self.final_act(out)
        return out