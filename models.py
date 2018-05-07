from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out, batch_norm=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, batch_norm=False):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels, batch_norm),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlockDropout(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True, batch_norm=False):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels, batch_norm),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels, batch_norm),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

'''
    VggUnet Architectures from https://github.com/ternaus/TernausNet/blob/master/unet_models.py
'''

class UNet11(nn.Module):
    def __init__(self, num_classes, num_filters=32, pretrained=False, batch_norm=False):
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            vgg - encoder pre-trained with VGG11
        :param batch_norm:
            False - no batchnormalization in encoder-decoder
            True - encoder: Conv-BN2d-ReLu
                   decoder: Conv-Relu-ConvTranspose2d-ReLu
        '''
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        if pretrained == 'vgg':
            if batch_norm:
                self.encoder = models.vgg11_bn(pretrained=True).features
            else:
                self.encoder = models.vgg11(pretrained=True).features
        else:
            if batch_norm:
                self.encoder = models.vgg11_bn(pretrained=False).features
            else:
                self.encoder = models.vgg11(pretrained=False).features

        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu
                                   )

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu
                                   )

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu
        )

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8, batch_norm)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8, batch_norm)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4, batch_norm)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2, batch_norm)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters, batch_norm)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters, batch_norm)
        

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))

class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False, batch_norm=False):
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        '''
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        if batch_norm:
            self.encoder = models.vgg16_bn(pretrained=pretrained).features
        else:
            self.encoder = models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        x_out = F.sigmoid(self.final(dec1))

        return x_out

class UNet11_dropout(nn.Module):
    def __init__(self, num_classes, num_filters=32, pretrained=False):
        '''
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            vgg - encoder pre-trained with VGG11
        '''
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        if pretrained == 'vgg':
            self.encoder = models.vgg11(pretrained=True).features
        else:
            self.encoder = models.vgg11(pretrained=False).features

        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   nn.Dropout2d(p=0.2))

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu,
                                   nn.Dropout2d(p=0.2))

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            nn.Dropout2d(p=0.2),
            self.encoder[8],
            self.relu,
            nn.Dropout2d(p=0.2)
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            nn.Dropout2d(p=0.2),
            self.encoder[13],
            self.relu,
            nn.Dropout2d(p=0.2)
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            nn.Dropout2d(p=0.2),
            self.encoder[18],
            self.relu,
            nn.Dropout2d(p=0.2),
        )

        self.center = DecoderBlockDropout(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlockDropout(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlockDropout(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlockDropout(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlockDropout(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))

'''
    Vanilla Unet from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
'''
def concat(xs):
    return torch.cat(xs, 1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

class UNet(nn.Module):
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.sigmoid(x_out)
        return x_out

def get_model(model_name, num_classes, pretrained=False, path='', **kwargs):
    model = model_name(num_classes, pretrained=pretrained, **kwargs)

    if pretrained == True:
        state = torch.load(path) # carvana path: TernausNet.pt
        model.load_state_dict(state['model'])
    
    return model
    