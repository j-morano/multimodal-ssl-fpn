import torch
import torch.nn as nn
import numpy as np

from config import config as global_config



class ModifiedUnet3D(nn.Module):
    """ReSensNet.
    ModifiedUnet3D U-Net architecture.
    Fully convolutional neural network with encoder/decoder architecture
    and skip connections, with dropout in the intermediate layer.

    Original input is: 16 x 64 x 1024 (Z x W x H)
    """

    def __init__(
        self,
        is_batchnorm=True,
        is_deconv=False,
        channels=[16, 32, 64, 128, 256],
        dropout=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        is_original=False,
    ):
        super().__init__()
        self.n_classes = global_config.num_outputs
        self.is_batchnorm = is_batchnorm
        self.in_channels = 1
        self.is_deconv = is_deconv
        self.use_1x1 = True
        self.is_original = is_original
        self.channels = channels
        self.dropout = dropout
        assert len(self.channels)==5
        assert len(self.dropout)==9

        print('Channel-variable: ' + str(self.channels))

        # DOWNsampling (done with either max-pooling, avg-pooling or
        #   3by3conv with stride of 2)
        self.conv1 = self._make_layer_2plus3(
            self.in_channels,
            self.channels[0],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[0]
        )
        self.conv2 = self._make_layer_2plus3(
            self.channels[0],
            self.channels[1],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[1]
        )
        self.conv3 = self._make_layer_2plus3(
            self.channels[1],
            self.channels[2],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[2]
        )
        self.conv4 = self._make_layer_2plus3(
            self.channels[2],
            self.channels[3],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[3]
        )
        self.conv5 = self._make_layer_2plus3(
            self.channels[3],
            self.channels[4],
            is_batchnorm = self.is_batchnorm,
            is_residual=True,
            dropout = self.dropout[4]
        )

        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2)) 
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2))

        if self.is_original:
            final_kernelsize = 8
        else:
            final_kernelsize = 4

        # NOTE: original has final_kernelsize=64
        self.zdimRed1 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[0],
            channels_out=self.channels[0],
            num_convreductions=4,
            final_kernelsize=final_kernelsize,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed2 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[1],
            channels_out=self.channels[1],
            num_convreductions=3,
            final_kernelsize=final_kernelsize,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed3 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[2],
            channels_out=self.channels[2],
            num_convreductions=2,
            final_kernelsize=final_kernelsize,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed4 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[3],
            channels_out=self.channels[3],
            num_convreductions=1,
            final_kernelsize=final_kernelsize,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )
        self.zdimRed5 = self._make_zdimReductionConvPlusFully(
            channels_in=self.channels[4],
            channels_out=self.channels[4],
            num_convreductions=0,
            final_kernelsize=final_kernelsize,
            is_batchnorm=self.is_batchnorm,
            is_residual=True,
            dropout=0.0
        )

        # UPsampling:
        self.up_concat4 = unet3dUp2modified(
            self.channels[4],
            self.channels[3],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[5],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat3 = unet3dUp2modified(
            self.channels[3],
            self.channels[2],
            upfactor=(2,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[6],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat2 = unet3dUp2modified(
            self.channels[2],
            self.channels[1],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[7],
            is_batchnorm=self.is_batchnorm
        )
        self.up_concat1 = unet3dUp2modified(
            self.channels[1],
            self.channels[0],
            upfactor=(1,2,1),
            is_deconv=self.is_deconv,
            is_residual=True,
            dropout=self.dropout[8],
            is_batchnorm=self.is_batchnorm
        )


        # final conv to reduce to one channel:
        # NOTE: original has nn.ReLU at the end
        self.final1 = nn.Sequential(
            nn.Conv3d(
                in_channels=self.channels[0],
                out_channels=self.n_classes,
                kernel_size=1
            )
            # nn.ReLU()
        )

    def _make_layer_2plus3(
        self,
        channels_in,
        channels_out,
        is_batchnorm,
        is_residual,
        dropout
    ):
        layers = []
        if channels_in == channels_out:
            downsample=None
        else:
            downsample = nn.Sequential(
                nn.Conv3d(
                    channels_in,
                    channels_out,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm3d(channels_out)
            )
        # two 1x3x3 kernels working within B-scans:
        layers.append(
            unet3dConvX(
                channels_in,
                channels_out,
                kernel_size=[(1,3,3),(1,3,3)],
                stride=[(1,1,1),(1,1,1)],
                padding=[(0,1,1),(0,1,1)],
                is_batchnorm=is_batchnorm,
                is_residual=is_residual,
                dropout=dropout,
                downsample=downsample
            )
        )
        # two 1x3x3 kernels working within B-scans + one 3x1x1 kernel working across B-scans:
        layers.append(
            unet3dConvX(
                channels_out,
                channels_out,
                kernel_size=[(1,3,3),(1,3,3),(3,1,1)],
                stride=[(1,1,1),(1,1,1),(1,1,1)],
                padding=[(0,1,1),(0,1,1),(1,0,0)],
                is_batchnorm=is_batchnorm,
                is_residual=is_residual,
                dropout=dropout,
                downsample=None
            )
        )
        return nn.Sequential(*layers)

    def _make_zdimReductionConvPlusFully(
        self,
        channels_in,
        channels_out,
        num_convreductions,
        final_kernelsize,
        is_batchnorm,
        is_residual,
        dropout
    ):
        layers=[]
        kernel_size=[]
        stride=[]
        padding=[]
        for _i in range(0,num_convreductions):
            kernel_size.append((1,1,3))
            stride.append((1,1,2))
            padding.append((0,0,1))

        if (channels_in != channels_out) or (num_convreductions>0 and is_residual):
            if is_batchnorm:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        channels_in,
                        channels_out,
                        kernel_size=(1,1,1),
                        stride=(1,1,2**(num_convreductions)),
                        bias=False
                    ),
                    nn.BatchNorm3d(channels_out)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        channels_in,
                        channels_out,
                        kernel_size=(1,1,1),
                        stride=(1,1,2**(num_convreductions)),
                        bias=True
                    )
                )
        else:
            downsample=None

        if num_convreductions>0:
            # X 1x1x3 kernels reducing the dimensionality of z:
            layers.append(
                unet3dConvX(
                    channels_in,
                    channels_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    is_batchnorm=is_batchnorm,
                    is_residual=is_residual,
                    dropout=dropout,
                    downsample=downsample
                )
            )
            # 1x1xN kernel reducing z-dimension to one:
            layers.append(
                unet3dConvX(
                    channels_out,
                    channels_out,
                    kernel_size=[(1,1,final_kernelsize)],
                    stride=[(1,1,1)],
                    padding=[(0,0,0)],
                    is_batchnorm=is_batchnorm,
                    is_residual=False,
                    dropout=dropout,
                    downsample=None
                )
            )
        else:
            # 1x1xN kernel reducing z-dimension to one:
            layers.append(
                unet3dConvX(
                    channels_in,
                    channels_out,
                    kernel_size=[(1,1,final_kernelsize)],
                    stride=[(1,1,1)],
                    padding=[(0,0,0)],
                    is_batchnorm=is_batchnorm,
                    is_residual=False,
                    dropout=dropout,
                    downsample=None
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Downsampling:
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        # NOTE: original has no means
        conv1 = self.zdimRed1(conv1)
        if not self.is_original:
            conv1 = torch.mean(conv1, dim=4, keepdim=True)
        conv2 = self.zdimRed2(conv2)
        if not self.is_original:
            conv2 = torch.mean(conv2, dim=4, keepdim=True)
        conv3 = self.zdimRed3(conv3)
        if not self.is_original:
            conv3 = torch.mean(conv3, dim=4, keepdim=True)
        conv4 = self.zdimRed4(conv4)
        if not self.is_original:
            conv4 = torch.mean(conv4, dim=4, keepdim=True)
        conv5 = self.zdimRed5(conv5)
        if not self.is_original:
            conv5 = torch.mean(conv5, dim=4, keepdim=True)

        # Upsampling:
        up4 = self.up_concat4(conv4, conv5)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # get the 'segmentation' using a 1x1 convolution
        if self.use_1x1:
            final_out = self.final1(up1)
        else:
            final_out = up1

        return final_out


class unet3dConvX(nn.Module):
    """Convolutional block with X convolutions in 3D
    Convolutional block:
        1.path: [X-1 times [Conv3d - Batch normalization - Relu] ]
                + [Conv3d - Batch normalization]
        2.path: identity
        then ReLU
    """

    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        is_batchnorm,
        is_residual,
        dropout,
        downsample
    ):
        super(unet3dConvX, self).__init__()

        layers = []
        for i in range(0,len(kernel_size)):
            if i==0 and i<len(kernel_size)-1:
                layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size),
                        nn.ReLU()
                    )
                )
            elif i==0 and i==len(kernel_size)-1:
                layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size)
                    )
                )
            elif 0<i<len(kernel_size)-1:
                layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[i],
                            stride=stride[i],
                            padding=padding[i],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size),
                        nn.ReLU()
                    )
                )
            elif i>0 and i==len(kernel_size)-1:
                layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=out_size,
                            out_channels=out_size,
                            kernel_size=kernel_size[len(kernel_size)-1],
                            stride=stride[len(kernel_size)-1],
                            padding=padding[len(kernel_size)-1],
                            bias=not(is_batchnorm)
                        ),
                        nn.BatchNorm3d(out_size)
                    )
                )
            else:
                raise AssertionError((
                    'UserException: in module "unet3dConvX".'
                    ' Error when value of iterator is "' + str(i)
                ))

        self.convBlock = nn.Sequential(*layers)

        self.is_residual = is_residual
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, x):
        residual = x

        out = self.convBlock(x)
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.is_residual:
            out += residual

        out = self.relu(out)

        if not (self.drop is None):
            out = self.drop(out)

        return out


class unet3dUp2modified(nn.Module):
    """Modified upsampling block for the 3D-Unet"""

    def __init__(
        self,
        lowlayer_channels,
        currlayer_channels,
        upfactor,
        is_deconv,
        is_residual,
        dropout,
        is_batchnorm
    ):
        super().__init__()
        # first an upsampling operation
        if is_deconv:
            self.up = nn.ConvTranspose3d(
                lowlayer_channels,
                currlayer_channels,
                kernel_size=upfactor,
                stride=upfactor
            )
        else:
            self.up = Upsample_Custom3d_nearest(
                scale_factor=upfactor,
                mode='nearest'
            )

        # and then a convolution
        if is_batchnorm:
            downsample = nn.Sequential(
                nn.Conv3d(
                    lowlayer_channels+currlayer_channels,
                    currlayer_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm3d(currlayer_channels))
        else:
            downsample = nn.Sequential(
                nn.Conv3d(
                    lowlayer_channels+currlayer_channels,
                    currlayer_channels,
                    kernel_size=1,
                    stride=1,
                    bias=True
                )
            )

        self.conv = unet3dConvX(
            in_size=lowlayer_channels+currlayer_channels,
            out_size=currlayer_channels,
            kernel_size=[(3,3,1),(3,3,1)],
            stride=[(1,1,1),(1,1,1)],
            padding=[(1,1,0),(1,1,0)],
            is_batchnorm=is_batchnorm,
            is_residual=is_residual,
            dropout=dropout,
            downsample=downsample
        )

    def forward(self, inputs1, inputs2):
        # upscale input coming from lower layer
        upsampled_inputs2 = self.up(inputs2)
        # convolution performed on concatenated inputs
        return self.conv(torch.cat([inputs1, upsampled_inputs2], 1))


class Upsample_Custom3d_nearest(nn.Module):
    """Upsamples a given multi-channel 3D (volumetric) data.

    Manual solution for non-uniform nearest-neighbor-upsampling.
    Simply make a new Variable that's the size needed and manually
    fill the array with data from the smaller variable. Idea taken from
    "https://github.com/pytorch/pytorch/issues/1487" and
    "https://gist.github.com/KeremTurgutlu/68feb119c9dd148285be2e247267a203"

    The input data is assumed to be of the form `minibatch x channels x
    depth x height x width`.
    Hence, for spatial inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor.

    One can give a :attr:`scale_factor` to calculate the output size.

    Args:
        scale_factor (a tuple of ints, optional): the multiplier for the
          image height / width / depth

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor(D_{in} * scale_factor[-3])` or `size[-3]`
          :math:`H_{out} = floor(H_{in} * scale_factor[-2])` or `size[-2]`
          :math:`W_{out} = floor(W_{in}  * scale_factor[-1])` or `size[-1]`
    """

    def __init__(self, scale_factor: list, mode='nearest'):
        super(Upsample_Custom3d_nearest, self).__init__()
        self.size = None
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        #depth wise interpolation
        depth_idx = (
            np.ceil(
                np.asarray( # type: ignore
                    list(range(1, 1 + int(input.shape[-3]*self.scale_factor[-3])))
                )/self.scale_factor[-3]
            ) - 1
        ).astype(int)
        # row wise interpolation
        row_idx =  (
            np.ceil(
                np.asarray( # type: ignore
                    list(range(1, 1 + int(input.shape[-2]*self.scale_factor[-2])))
                )/self.scale_factor[-2]
            ) - 1
        ).astype(int)
        # column wise interpolation
        col_idx = (
            np.ceil(
                np.asarray( # type: ignore
                    list(range(1, 1 + int(input.shape[-1]*self.scale_factor[-1])))
                )/self.scale_factor[-1]
            ) - 1
        ).astype(int)

        # Create nearest-neighbor upsampled matrix and return it:
        return input[:,:,depth_idx,:,:][:,:,:,row_idx,:][:,:,:,:,col_idx]

    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return self.__class__.__name__ + '(' + info + ')'
