import torch.utils.data
from tqdm import tqdm

from torch.amp import autocast
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostModule3D, self).__init__()
        self.oup = out_channels
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)

        # Primary standard convolution + BN + ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Secondary depthwise convolution + BN + ReLU
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(
                in_channels=out_channels // 2,
                out_channels=out_channels // 2,
                kernel_size=kernel_size,
                padding=padding,
                groups=out_channels // 2,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, ghost=False, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        if ghost:
            self.double_conv = nn.Sequential(
                GhostModule3D(in_channels, out_channels, kernel_size),
                GhostModule3D(out_channels, out_channels, kernel_size),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="replicate",
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode="replicate",
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels, ghost=False, depth_down=False, kernel_size=(3, 3, 3)):
        super().__init__()
        if depth_down:
            kernel_depth = 2
        else:
            kernel_depth = 1
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(
                # 缩小宽高，根据参数可以选择缩小深度
                # 如果输入时长宽不能被2整除，那么最终的长宽是向下取整的
                kernel_size=(kernel_depth, 2, 2),
            ),
            DoubleConv3D(in_channels, out_channels, ghost, kernel_size),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, ghost=False, depth_up=False, kernel_size=(3, 3, 3)):
        super().__init__()
        if depth_up:
            kernel_depth = 2
        else:
            kernel_depth = 1
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            # 根据参数可以选择增加深度
            kernel_size=(kernel_depth, 2, 2),
            stride=(kernel_depth, 2, 2),
        )
        self.conv = DoubleConv3D(in_channels, out_channels, ghost, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x2.size()[-2] < x1.size()[-2] or x2.size()[-1] < x1.size()[-1] or x2.size()[-3] < x1.size()[-3]:
            x1, x2 = x2, x1
        diffD = x2.size()[-3] - x1.size()[-3]
        diffH = x2.size()[-2] - x1.size()[-2]
        diffW = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(
            x1,
            [
                diffW // 2,
                diffW - diffW // 2,  # 左右
                diffH // 2,
                diffH - diffH // 2,  # 上下
                diffD // 2,
                diffD - diffD // 2,
            ],
            mode="replicate",
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Merge3D(nn.Module):
    def __init__(self, scale):
        super(Merge3D, self).__init__()
        self.scale = scale
        self.scaleup = torch.nn.Upsample(scale_factor=(1, self.scale, self.scale))

    def forward(self, x1, x2):
        x2 = self.scaleup(x2)
        if x2.size()[-2] < x1.size()[-2] or x2.size()[-1] < x1.size()[-1] or x2.size()[-3] < x1.size()[-3]:
            x1, x2 = x2, x1
        diffD = x2.size()[-3] - x1.size()[-3]
        diffH = x2.size()[-2] - x1.size()[-2]
        diffW = x2.size()[-1] - x1.size()[-1]
        x2 = F.pad(
            x2,
            [
                diffW // 2,
                diffW - diffW // 2,  # 左右
                diffH // 2,
                diffH - diffH // 2,  # 上下
                diffD // 2,
                diffD - diffD // 2,
            ],
            mode="replicate",
        )
        return torch.add(x1, x2)


class ConvUNeXtCB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cbd1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=5,
                padding=2,
                padding_mode="replicate",
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.Conv3d(
                in_channels=channels,
                out_channels=4 * channels,
                kernel_size=1,
                padding=0,
                padding_mode="replicate",
                bias=False,
            ),
            nn.Conv3d(
                in_channels=4 * channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
                padding_mode="replicate",
                bias=False,
            ),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x_1 = self.cbd1(x)
        x = torch.add(x_1, x)
        x = self.gelu(x)
        return x


class ConvUNeXtDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.DS = nn.Sequential(
            # nn.Conv3d(
            #    in_channels=in_channels, out_channels=out_channels,
            #    kernel_size=2, stride=2, bias=False
            # ),
            # nn.BatchNorm3d(out_channels)
            nn.MaxPool3d(kernel_size=1),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.DS(x)


class ConvUNeXtAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm3d(channels)
        self.ups = nn.Upsample(scale_factor=2)
        self.up_c = nn.Conv3d(in_channels=channels, out_channels=channels // 2, kernel_size=1)
        self.linear = nn.Linear(channels // 2, channels * 3 // 2, False)
        self.linear_c1 = nn.Linear(channels // 2, channels // 2, False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_final = nn.Linear(channels // 2, channels // 2, False)
        self.channel_correct = nn.Conv3d(in_channels=channels, out_channels=channels // 2, kernel_size=1, bias=False)

    def forward(self, x1, x2):
        c = self.bn(x1)
        c = self.ups(c)
        c = self.up_c(c)
        diffD = x2.size()[-3] - c.size()[-3]
        diffH = x2.size()[-2] - c.size()[-2]
        diffW = x2.size()[-1] - c.size()[-1]
        c = F.pad(
            c,
            [
                diffW // 2,
                diffW - diffW // 2,  # 左右
                diffH // 2,
                diffH - diffH // 2,  # 上下
                diffD // 2,
                diffD - diffD // 2,
            ],
            mode="replicate",
        )
        # Who designed the nn.linear??? Why does it need C to be the last???
        # Original: B, C, D, H, W
        # Required: B, D, H, W, C
        c = torch.permute(c, (0, 2, 3, 4, 1))
        c = self.linear(c)
        c = torch.permute(c, (0, 4, 1, 2, 3))
        c = torch.chunk(c, chunks=3, dim=1)
        c1, c2, c3 = c[0], c[1], c[2]
        # c1
        c1 = torch.permute(c1, (0, 2, 3, 4, 1))
        c1 = self.linear_c1(c1)
        c1 = torch.permute(c1, (0, 4, 1, 2, 3))
        c1 = torch.add(c1, x2)
        c1 = self.sigmoid(c1)
        c1 = torch.mul(c1, x2)
        # c2 and c3
        c2 = self.sigmoid(c2)
        c3 = self.tanh(c3)
        c23 = torch.mul(c2, c3)
        # final
        x = torch.add(c1, c23)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = self.linear_final(x)
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = torch.cat([x, x2], dim=1)
        x = self.channel_correct(x)
        return x


class HalfNet(nn.Module):
    def __init__(self):
        super(HalfNet, self).__init__()
        channel_base = 8
        self.inc = DoubleConv3D(1, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.down1 = Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge1 = Merge3D(2)
        self.down2 = Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge2 = Merge3D(4)
        self.down3 = Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge3 = Merge3D(8)
        self.down4 = Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge4 = Merge3D(16)
        self.down5 = Down3D(channel_base, channel_base, ghost=False, kernel_size=(1, 3, 3))
        self.merge5 = Merge3D(32)
        self.z_work = DoubleConv3D(channel_base, channel_base, ghost=False, kernel_size=(3, 1, 1))
        self.outc = OutConv3D(channel_base, 2)

    def forward(self, x):
        x = self.inc(x)
        x2 = self.down1(x)
        x = self.merge1(x, x2)
        x2 = self.down2(x2)
        x = self.merge2(x, x2)
        x2 = self.down3(x2)
        x = self.merge3(x, x2)
        x2 = self.down4(x2)
        x = self.merge4(x, x2)
        x2 = self.down5(x2)
        x = self.merge5(x, x2)
        x = self.z_work(x)
        x = self.outc(x)
        return x


def dummy_train(use_autocast=True, device=torch.device("cpu")):
    train_dataset = torch.utils.data.TensorDataset(
        torch.rand(2, 1, 5, 512, 512),
        torch.zeros(2, 5, 512, 512, dtype=torch.int64),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True
    )

    model = HalfNet().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())

    ctx_manager = autocast(device.type) if use_autocast else nullcontext()

    for epoch in range(100):
        for i, batch in tqdm(enumerate(train_loader)):
            x, y = batch
            x, y = x.to(device), y.to(device)

            with ctx_manager:
                y_hat = model(x.float())
                loss = F.cross_entropy(input=y_hat, target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # Compare the CPU usage between these two:
    # dummy_train(use_autocast=True)
    dummy_train(use_autocast=False)

    # And the GPU version:
    # dummy_train(use_autocast=True, device=torch.device("cuda", 0))
    # dummy_train(use_autocast=False, device=torch.device("cuda", 0))
