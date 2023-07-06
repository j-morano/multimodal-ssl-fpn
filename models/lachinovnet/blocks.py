import torch.nn as nn
import torch.nn.functional as F



class Trilinear(nn.Module):
    def __init__(self, scale):
        super(Trilinear, self).__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x,scale_factor=self.scale,mode='trilinear')
        return out


class Residual_bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        downsample = None,
        kernel_size = None,
        padding = None
    ):
        super(Residual_bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4
        #groups = 8

        if kernel_size is None:
            kernel_size = (1,3,1)
            padding = (0,1,0)

        self.conv_pre = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels//compression,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        assert padding is not None
        self.conv1 = nn.Conv3d(
            in_channels=out_channels//compression,
            out_channels=out_channels//compression,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            groups=1
        )

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(
            in_channels=out_channels // compression,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out


class Residual_bottleneck_bn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        downsample = None,
        kernel_size = None,
        padding = None
    ):
        super(Residual_bottleneck_bn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 8
        #groups = 8

        if kernel_size is None:
            kernel_size = (1,3,3)
            padding = (0,1,1)

        self.conv_pre = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels//compression,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False
        )
        assert padding is not None
        self.conv1 = nn.Conv3d(
            in_channels=out_channels//compression,
            out_channels=out_channels//compression,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            groups=1
        )

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(
            in_channels=out_channels // compression,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out
