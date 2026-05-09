import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(dropout)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)

        return out

class ResUNetOrdinal(nn.Module):
    """
    Baseline UNet: flatten all timesteps into channels.
    
    Input:  [B, T, C, H, W]
    Flatten: [B, T*C, H, W]
    Output: [B, num_classes, H, W]
    """
    def __init__(
        self,
        in_channels  = 13,
        num_timesteps= 34,
        num_classes  = 4,
        base_dim     = 64,
        dropout      = 0.2,
    ):
        super().__init__()
        flat_channels = in_channels * num_timesteps  # 13*34 = 442

        # -------- Encoder (3 levels) --------
        self.enc1 = ResBlock(flat_channels, base_dim, dropout)
        self.enc2 = ResBlock(base_dim, base_dim * 2, dropout)
        self.enc3 = ResBlock(base_dim * 2, base_dim * 4, dropout)

        self.pool = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = ResBlock(base_dim * 4, base_dim * 8, dropout)

        # -------- Decoder --------
        self.up3  = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, 2, stride=2)
        self.dec3 = ResBlock(base_dim * 8, base_dim * 4, dropout)

        self.up2  = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.dec2 = ResBlock(base_dim * 4, base_dim * 2, dropout)

        self.up1  = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.dec1 = ResBlock(base_dim * 2, base_dim, dropout)

        self.head = nn.Conv2d(base_dim, 1, kernel_size=1)

        # K-1 thresholds
        self.raw_bias = nn.Parameter(torch.zeros(num_classes))

    def _block(self, in_ch, out_ch, dropout=0.2):
        return ResBlock(in_ch, out_ch, dropout)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # -------- Bottleneck --------
        b = self.bottleneck(self.pool(e3))

        # -------- Decoder --------
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # -------- CORAL Head --------
        features = self.head(d1)  # [B,1,H,W]

        thresholds = torch.cumsum(F.softplus(self.raw_bias), dim=0)
        thresholds = thresholds.view(1, -1, 1, 1)

        logits = features - thresholds  # [B, K-1, H, W]

        return logits
       