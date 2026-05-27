import torch
import torch.nn as nn
from torchinfo import summary

# Noise2Noise U-Net
class Noise2NoiseUNet(nn.Module):
    # Helpers
    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # The final layer in the paper is linear, which expects a gain of 1.0.
                if name == "dec_conv1c":
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _conv(self, in_ch: int, out_ch: int) -> nn.Conv2d:
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)         

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            self._conv(in_ch, out_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()
        n, m = in_channels, out_channels

        # Encoder
        self.enc_conv0 = self._conv_block(n, 48)   # INPUT  → 48
        self.enc_conv1 = self._conv_block(48, 48)   # 48     → 48
        self.pool1     = nn.MaxPool2d(2)

        self.enc_conv2 = self._conv_block(48, 48)   # 48     → 48
        self.pool2     = nn.MaxPool2d(2)

        self.enc_conv3 = self._conv_block(48, 48)   # 48     → 48
        self.pool3     = nn.MaxPool2d(2)

        self.enc_conv4 = self._conv_block(48, 48)   # 48     → 48
        self.pool4     = nn.MaxPool2d(2)

        self.enc_conv5 = self._conv_block(48, 48)   # 48     → 48
        self.pool5     = nn.MaxPool2d(2)

        # Bottleneck
        self.enc_conv6 = self._conv_block(48, 48)   # 48     → 48

        # Decoder
        # Level 5: upsample → concat with pool4 (48) → 96
        self.upsample5  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv5a = self._conv_block(96, 96)
        self.dec_conv5b = self._conv_block(96, 96)

        # Level 4: upsample → concat with pool3 (48) → 144
        self.upsample4  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv4a = self._conv_block(144, 96)
        self.dec_conv4b = self._conv_block(96, 96)

        # Level 3: upsample → concat with pool2 (48) → 144
        self.upsample3  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv3a = self._conv_block(144, 96)
        self.dec_conv3b = self._conv_block(96, 96)

        # Level 2: upsample → concat with pool1 (48) → 144
        self.upsample2  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv2a = self._conv_block(144, 96)
        self.dec_conv2b = self._conv_block(96, 96)

        # Level 1: upsample → concat with input (n) → 96+n
        self.upsample1  = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv1a = self._conv_block(96 + n, 64)
        self.dec_conv1b = self._conv_block(64, 32)
        # Final output conv — linear activation (no non-linearity)
        self.dec_conv1c = self._conv(32, m)

        # Weight initialisation
        self._init_weights()

    # Forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = x                             # keep raw input for CONCAT1

        e0 = self.enc_conv0(x0)            # (B, 48, H,    W)
        e1 = self.enc_conv1(e0)            # (B, 48, H,    W)
        p1 = self.pool1(e1)                # (B, 48, H/2,  W/2)  — skip for CONCAT2

        e2 = self.enc_conv2(p1)            # (B, 48, H/2,  W/2)
        p2 = self.pool2(e2)                # (B, 48, H/4,  W/4)  — skip for CONCAT3

        e3 = self.enc_conv3(p2)            # (B, 48, H/4,  W/4)
        p3 = self.pool3(e3)                # (B, 48, H/8,  W/8)  — skip for CONCAT4

        e4 = self.enc_conv4(p3)            # (B, 48, H/8,  W/8)
        p4 = self.pool4(e4)                # (B, 48, H/16, W/16) — skip for CONCAT5

        e5 = self.enc_conv5(p4)            # (B, 48, H/16, W/16)
        p5 = self.pool5(e5)                # (B, 48, H/32, W/32)

        # Bottleneck
        e6 = self.enc_conv6(p5)            # (B, 48, H/32, W/32)

        # Decoder
        # Level 5
        d5 = self.upsample5(e6)            # (B, 48,  H/16, W/16)
        d5 = torch.cat([d5, p4], dim=1)    # (B, 96,  H/16, W/16)  — CONCAT5
        d5 = self.dec_conv5a(d5)           # (B, 96,  H/16, W/16)
        d5 = self.dec_conv5b(d5)           # (B, 96,  H/16, W/16)

        # Level 4
        d4 = self.upsample4(d5)            # (B, 96,  H/8,  W/8)
        d4 = torch.cat([d4, p3], dim=1)    # (B, 144, H/8,  W/8)   — CONCAT4
        d4 = self.dec_conv4a(d4)           # (B, 96,  H/8,  W/8)
        d4 = self.dec_conv4b(d4)           # (B, 96,  H/8,  W/8)

        # Level 3
        d3 = self.upsample3(d4)            # (B, 96,  H/4,  W/4)
        d3 = torch.cat([d3, p2], dim=1)    # (B, 144, H/4,  W/4)   — CONCAT3
        d3 = self.dec_conv3a(d3)           # (B, 96,  H/4,  W/4)
        d3 = self.dec_conv3b(d3)           # (B, 96,  H/4,  W/4)

        # Level 2
        d2 = self.upsample2(d3)            # (B, 96,  H/2,  W/2)
        d2 = torch.cat([d2, p1], dim=1)    # (B, 144, H/2,  W/2)   — CONCAT2
        d2 = self.dec_conv2a(d2)           # (B, 96,  H/2,  W/2)
        d2 = self.dec_conv2b(d2)           # (B, 96,  H/2,  W/2)

        # Level 1
        d1 = self.upsample1(d2)            # (B, 96,  H,    W)
        d1 = torch.cat([d1, x0], dim=1)    # (B, 96+n, H,   W)     — CONCAT1
        d1 = self.dec_conv1a(d1)           # (B, 64,  H,    W)
        d1 = self.dec_conv1b(d1)           # (B, 32,  H,    W)
        out = self.dec_conv1c(d1)          # (B, m,   H,    W)  — linear

        return out

def noise2noise_rgb() -> Noise2NoiseUNet:
    """Gaussian / Poisson / Bernoulli / text-removal (n=3, m=3)."""
    return Noise2NoiseUNet(in_channels=3, out_channels=3)


def noise2noise_mc() -> Noise2NoiseUNet:
    """Monte Carlo denoising: RGB + albedo + normal → RGB (n=9, m=3)."""
    return Noise2NoiseUNet(in_channels=9, out_channels=3)


def noise2noise_mri() -> Noise2NoiseUNet:
    """MRI reconstruction: monochrome → monochrome (n=1, m=1)."""
    return Noise2NoiseUNet(in_channels=1, out_channels=1)


# Quick parameter / shape summary (not a training loop)

if __name__ == "__main__":
    import sys

    configs = [
        ("RGB denoising (n=3, m=3)",        noise2noise_rgb,  (1, 3, 256, 256)),
        # ("Monte Carlo denoising (n=9, m=3)", noise2noise_mc,   (1, 9, 256, 256)),
        # ("MRI reconstruction (n=1, m=1)",    noise2noise_mri,  (1, 1, 256, 256)),
    ]

    for name, factory, shape in configs:
        model = factory()
        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.zeros(*shape)
        with torch.no_grad():
            out = model(dummy)
        print(f"{name}")
        print(f"  Parameters : {n_params:,}")
        print(f"  Input  shape: {tuple(dummy.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        print()
        
        summary(model)

        print(model)