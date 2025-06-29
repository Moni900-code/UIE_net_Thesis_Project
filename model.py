import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# CBAM Module
# -------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        channel_att = self.channel_gate(x)
        x = x * channel_att
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_gate(spatial_input)
        spatial_att = torch.sigmoid(spatial_att)
        x = x * spatial_att
        return x

# -------------------------
# ConvBlock
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ConvBlock, self).__init__()
        self.use_cbam = use_cbam
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs = nn.Hardswish()
        if self.use_cbam:
            self.cbam = CBAM(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.hs(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.pw(x)
        x = self.bn2(x)
        return x

# -------------------------
# Trainable Color Estimator
# -------------------------
class ColorEstimator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ColorEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# Color Recovery Module
# -------------------------
class ColorRecoveryModule(nn.Module):
    def __init__(self, in_channels):
        super(ColorRecoveryModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, content_features, color_features):
        D = -content_features - color_features
        M = content_features * color_features
        L = 2 * torch.sigmoid(D) * torch.tanh(M)
        sigmoid_D = torch.sigmoid(D)
        sigmoid_D = torch.clamp(sigmoid_D, 0, 0.5)
        L = torch.clamp(L, 0, 1)

        output_features = []
        current_color = color_features
        for _ in range(4):
            F_i = L * current_color + content_features
            output_features.append(F_i)
            current_color = self.conv1x1(F_i)
            current_color = F.relu(current_color)

        return torch.mean(torch.stack(output_features), dim=0)

# -------------------------
# Mynet with Trainable Color Estimator
# -------------------------
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(3, 16, kernel_size=1, bias=False)
        self.bn_input = nn.BatchNorm2d(16)
        self.hs_input = nn.Hardswish()

        self.block1 = ConvBlock(16, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(80, 32, use_cbam=True)

        # Color Estimator: learns to estimate GT-like color from degraded input
        self.color_estimator = ColorEstimator(in_channels=3, out_channels=32)
        self.crm = ColorRecoveryModule(in_channels=32)

        self.output = nn.Conv2d(32, 3, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x, gt_color_source=None):
        # Estimate color feature from degraded image
        color_features = self.color_estimator(x)

        # Optional: supervise estimator with GT color image
        color_supervision_loss = None
        if self.training and gt_color_source is not None:
            with torch.no_grad():
                gt_color_features = self.color_estimator(gt_color_source)
            color_supervision_loss = F.mse_loss(color_features, gt_color_features)

        # Content path
        x = self.input(x)
        x = self.bn_input(x)
        x = self.hs_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.cat([x, torch.zeros_like(x)[:, :16, :, :]], dim=1)
        content_features = self.block3(x)

        # Match resolution
        color_features = F.interpolate(color_features, size=content_features.shape[2:], mode='bilinear', align_corners=False)

        # CRM Fusion
        x = self.crm(content_features, color_features)
        x = self.output(x)
        x = self.final_act(x)

        return x, color_supervision_loss
