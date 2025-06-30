import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from ptflops import get_model_complexity_info

# -------------------------
# CBAM Module
# -------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Hardsigmoid()
        )
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        channel_att = self.channel_gate(x)
        x = x * channel_att
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = torch.sigmoid(self.spatial_gate(spatial_input))
        x = x * spatial_att
        return x

# -------------------------
# ConvBlock
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ConvBlock, self).__init__()
        self.use_cbam = use_cbam
        self.dw = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.hs = nn.Hardswish()
        if use_cbam:
            self.cbam = CBAM(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.hs(self.bn1(self.dw(x)))
        if self.use_cbam:
            x = self.cbam(x)
        return self.bn2(self.pw(x))

# -------------------------
# Color Feature Extractor
# -------------------------
class ColorFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ColorFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(16, out_channels, 1, bias=False)

    def forward(self, x):
        return self.decoder(self.encoder(x))

# -------------------------
# Balanced Stretching Module
# -------------------------
class BalancedStretchingModule(nn.Module):
    def __init__(self, in_channels):
        super(BalancedStretchingModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        Ay = self.conv1x1(torch.mean(x, dim=(2, 3), keepdim=True))
        My = self.conv1x1(torch.max(x.view(x.size(0), x.size(1), -1), dim=2)[0].view(x.size(0), x.size(1), 1, 1))
        x = self.bn(x + Ay + My)
        return x + F.relu(self.conv3x3(x))

# -------------------------
# Efficient Transformer-based PLM
# -------------------------
class PredictionLearningModule(nn.Module):
    def __init__(self, in_channels, num_heads=4, reduction=4):
        super(PredictionLearningModule, self).__init__()
        self.reduction = reduction
        self.down = nn.AvgPool2d(reduction)
        self.up = nn.Upsample(scale_factor=reduction, mode='bilinear', align_corners=False)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 1)
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.alpha = 1e-8

    def forward(self, x):
        res = x
        x = F.relu(self.bn(self.conv3x3(x)))
        x_ds = self.down(x)
        b, c, h, w = x_ds.shape
        x_flat = x_ds.flatten(2).permute(0, 2, 1)
        q = torch.zeros_like(x_flat)
        x_attn, _ = self.attn(q, x_flat, x_flat)
        x_attn = x_attn.permute(0, 2, 1).view(b, c, h, w)
        x_attn = self.up(self.ffn(x_attn))
        return res + self.gamma * torch.clamp(x_attn, min=self.alpha)

# -------------------------
# Color Recovery Module
# -------------------------
class ColorRecoveryModule(nn.Module):
    def __init__(self, in_channels):
        super(ColorRecoveryModule, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, content, color):
        D = -content - color
        M = content * color
        L = 2 * torch.sigmoid(D) * torch.tanh(M)
        L = torch.clamp(L, 0, 1)
        current = color
        outputs = []
        for _ in range(4):
            F_i = L * current + content
            outputs.append(F_i)
            current = F.relu(self.conv1x1(F_i))
        return torch.mean(torch.stack(outputs), dim=0)

# -------------------------
# Main Network
# -------------------------
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        self.block1 = ConvBlock(16, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(80, 32, use_cbam=True)
        self.color_extractor = ColorFeatureExtractor(3, 32)
        self.bsm = BalancedStretchingModule(32)
        self.crm = ColorRecoveryModule(32)
        self.plm = PredictionLearningModule(32, num_heads=4)
        self.output = nn.Conv2d(32, 3, 1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        color = self.color_extractor(x)
        x = self.input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = F.pad(x, (0, 0, 0, 0, 0, 16))  # Pad to 80 channels
        content = self.block3(x)
        content = self.bsm(content)
        color = F.interpolate(color, size=content.shape[2:], mode='bilinear', align_corners=False)
        x = self.crm(content, color)
        x = self.plm(x)
        return self.final_act(self.output(x))

# -------------------------
# Main: Summary + FLOPs
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mynet().to(device)

    print("\n Model Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Total trainable parameters: {total_params}")

    print("\n Model Summary:")
    summary(model, input_size=(3, 224, 224))

    print("\n Calculating FLOPs:")
    with torch.cuda.device(0 if torch.cuda.is_available() else "cpu"):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"\n FLOPs: {macs}")
        print(f" Parameters: {params}")
