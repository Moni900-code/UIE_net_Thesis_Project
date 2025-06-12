import torch
import torch.nn as nn
from torchsummary import summary
from ptflops import get_model_complexity_info

# -------------------------
# CBAM Module
# -------------------------
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):  # Increased reduction ratio for lighter CBAM
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, bias=False),
            nn.Hardsigmoid()
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=2, bias=False)  # Reduced kernel size

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
# Mynet Model
# -------------------------
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.input = nn.Conv2d(3, 8, kernel_size=1, stride=1, bias=False)  # Reduced channels
        self.bn_input = nn.BatchNorm2d(8)
        self.hs_input = nn.Hardswish()
        
        self.block1 = ConvBlock(8, 16, stride=1)  # Reduced channels
        self.block2 = ConvBlock(16, 32, stride=1)  # Reduced channels
        self.block3 = ConvBlock(48, 16, stride=1, use_cbam=True)  # Reduced channels, kept CBAM
        
        self.output = nn.Conv2d(16, 3, kernel_size=1, stride=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.bn_input(x)
        x = self.hs_input(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = torch.cat([x, torch.zeros_like(x)[:, :16, :, :]], dim=1)  # Pad to 48 channels
        x = self.block3(x)
        
        x = self.output(x)
        x = self.final_act(x)
        return x

# -------------------------
# Main: Summary + FLOPs
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mynet().to(device)

    print("\n✅ Model Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔢 Total trainable parameters: {total_params}")

    print("\n📊 Model Summary:")
    summary(model, input_size=(3, 224, 224))

    print("\n🧮 Calculating FLOPs:")
    with torch.cuda.device(0 if torch.cuda.is_available() else "cpu"):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"\n🔥 FLOPs: {macs}")
        print(f"💾 Parameters: {params}")
