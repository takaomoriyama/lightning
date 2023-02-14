# 包括实现UNet需要的nn模组
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====3D=====

# https://blog.paperspace.com/ghostnet-cvpr-2020/
# 可以替代标准的卷积，理论上速度更快
class GhostModule3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3)):
        super(GhostModule3D, self).__init__()
        self.oup = out_channels
        padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2)

        # Primary standard convolution + BN + ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels // 2,
                      kernel_size=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Secondary depthwise convolution + BN + ReLU
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(in_channels=out_channels // 2,
                      out_channels=out_channels // 2,
                      kernel_size=kernel_size,
                      padding=padding,
                      groups=out_channels // 2,
                      bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


# 两次3维卷积，可选用Ghost版本来加速
class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, ghost=False, kernel_size=(3, 3, 3)):
        super().__init__()
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        if ghost:
            self.double_conv = nn.Sequential(
                GhostModule3D(in_channels, out_channels, kernel_size),
                GhostModule3D(out_channels, out_channels, kernel_size)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                    bias=False
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=padding, padding_mode='replicate',
                    bias=False
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


# 一次Max Pooling后接两次3维卷积
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
            DoubleConv3D(in_channels, out_channels, ghost, kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 一次transposed convolution，进行Concatenates，然后再接两次3维卷积
class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, ghost=False, depth_up=False, kernel_size=(3, 3, 3)):
        super().__init__()
        if depth_up:
            kernel_depth = 2
        else:
            kernel_depth = 1
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=in_channels // 2,
            # 根据参数可以选择增加深度
            kernel_size=(kernel_depth, 2, 2), stride=(kernel_depth, 2, 2)
        )
        self.conv = DoubleConv3D(in_channels, out_channels, ghost, kernel_size)

    def forward(self, x1, x2):
        # 将输入进行transposed convolution
        x1 = self.up(x1)
        # 图像：(N,C,D,H,W) N=Batch C=Channel
        # 理论上x2一定比x1相等或更大，但为了增加适用性，只要x2有一个方向的尺寸大于x1，就会将x1和x2调换
        if x2.size()[-2] < x1.size()[-2] or x2.size()[-1] < x1.size()[-1] or x2.size()[-3] < x1.size()[-3]:
            x1, x2 = x2, x1
        # 找到图像在深度上的区别
        diffD = x2.size()[-3] - x1.size()[-3]
        # 找到图像在高度上的区别
        diffH = x2.size()[-2] - x1.size()[-2]
        # 找到图像在宽度上的区别
        diffW = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,  # 左右
                        diffH // 2, diffH - diffH // 2,  # 上下
                        diffD // 2, diffD - diffD // 2],
                   mode="replicate")
        # 将x1和x2的channel加起来
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 进行单次1x1卷积，用于最终输出，不提供Ghost版本
class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# 将来自不同层的结果相加起来，用于HalfNet，学名是feature fusion
class Merge3D(nn.Module):
    def __init__(self, scale):
        super(Merge3D, self).__init__()
        self.scale = scale
        self.scaleup = torch.nn.Upsample(scale_factor=(1, self.scale, self.scale))

    def forward(self, x1, x2):
        # x2的尺寸比较小，所以需要放大
        x2 = self.scaleup(x2)
        if x2.size()[-2] < x1.size()[-2] or x2.size()[-1] < x1.size()[-1] or x2.size()[-3] < x1.size()[-3]:
            x1, x2 = x2, x1
        diffD = x2.size()[-3] - x1.size()[-3]
        diffH = x2.size()[-2] - x1.size()[-2]
        diffW = x2.size()[-1] - x1.size()[-1]
        x2 = F.pad(x2, [diffW // 2, diffW - diffW // 2,  # 左右
                        diffH // 2, diffH - diffH // 2,  # 上下
                        diffD // 2, diffD - diffD // 2],
                   mode="replicate")
        return torch.add(x1, x2)


# ConvUNeXt Convolution blocks
class ConvUNeXtCB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cbd1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels, out_channels=channels,
                kernel_size=5, padding=2, padding_mode='replicate',
                bias=False
            ),
            nn.BatchNorm3d(channels),
            nn.Conv3d(
                in_channels=channels, out_channels=4 * channels,
                kernel_size=1, padding=0, padding_mode='replicate',
                bias=False
            ),
            nn.Conv3d(
                in_channels=4 * channels, out_channels=channels,
                kernel_size=1, padding=0, padding_mode='replicate',
                bias=False
            )
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        x_1 = self.cbd1(x)
        x = torch.add(x_1, x)
        x = self.gelu(x)
        return x


# ConvUNeXt down-sampling
class ConvUNeXtDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.DS = nn.Sequential(
            #nn.Conv3d(
            #    in_channels=in_channels, out_channels=out_channels,
            #    kernel_size=2, stride=2, bias=False
            #),
            #nn.BatchNorm3d(out_channels)
            nn.MaxPool3d(kernel_size=1),
            nn.Conv3d(
               in_channels=in_channels, out_channels=out_channels,
               kernel_size=1
            ),
        )

    def forward(self, x):
        return self.DS(x)




# ConvUNeXt Attention Gate
class ConvUNeXtAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm3d(channels)
        self.ups = nn.Upsample(scale_factor=2)
        self.up_c = nn.Conv3d(in_channels=channels, out_channels=channels//2, kernel_size=1)
        self.linear = nn.Linear(channels//2, channels * 3 // 2, False)
        self.linear_c1 = nn.Linear(channels//2, channels//2, False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.linear_final = nn.Linear(channels//2, channels//2, False)
        self.channel_correct = nn.Conv3d(
            in_channels=channels, out_channels=channels//2,
            kernel_size=1, bias=False
        )

    def forward(self, x1, x2):
        c = self.bn(x1)
        c = self.ups(c)
        c = self.up_c(c)
        diffD = x2.size()[-3] - c.size()[-3]
        diffH = x2.size()[-2] - c.size()[-2]
        diffW = x2.size()[-1] - c.size()[-1]
        c = F.pad(c, [diffW // 2, diffW - diffW // 2,  # 左右
                      diffH // 2, diffH - diffH // 2,  # 上下
                      diffD // 2, diffD - diffD // 2],
                  mode="replicate")
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


# 这里的函数用于测试各个组件是否能正常运作
if __name__ == "__main__":
    #train_data = DataComponents.Train_Val_Dataset('datasets/train/img', 'datasets/train/lab')
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=8)
    test_tensor_1 = torch.randn(1, 32, 10, 1024, 1024)
    test_tensor_2 = torch.randn(1, 512, 10, 128, 128)
    #print(test_tensor_1.shape)
    test_conv = GhostModule3D(32, 32)
    test_tensor_3 = test_conv.forward(test_tensor_1)
    print(test_tensor_3.shape)
