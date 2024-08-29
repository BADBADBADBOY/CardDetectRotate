from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rep_blocks import hswish, hsigmoid

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = nn.BatchNorm2d(
            num_filters)
        self.hardswish = hswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False
                 ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            )
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = hsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.mul(identity,x)
        return x


class PPLCNet(nn.Module):
    def __init__(self,
                 heads,
                 scale=0.5,
                 head_conv=64,
                 stride_list=[2, 2, 2, 2, 2],
                 use_last_conv=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.heads = heads
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG
        self.inplanes = make_divisible(512*scale)
        assert isinstance(self.stride_list, (
            list, tuple
        )), "stride_list should be in (list, tuple) but got {}".format(
            type(self.stride_list))
        assert len(self.stride_list
                   ) == 5, "stride_list length should be 5 but got {}".format(
                       len(self.stride_list))

        for i, stride in enumerate(stride_list[1:]):
            self.net_config["blocks{}".format(i + 3)][0][3] = stride
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=stride_list[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se
                    ) in enumerate(self.net_config["blocks6"])
        ])
        self.adaption3 = nn.Conv2d(
            make_divisible(256*self.scale), make_divisible(256*self.scale), kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption2 = nn.Conv2d(
            make_divisible(128*self.scale), make_divisible(256*self.scale), kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption1 = nn.Conv2d(
            make_divisible(64*self.scale), make_divisible(256*self.scale), kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption0 = nn.Conv2d(
            make_divisible(32*self.scale), make_divisible(256*self.scale), kernel_size=1, stride=1, padding=0, bias=False)

        self.adaptionU1 = nn.Conv2d(
            make_divisible(256*self.scale), make_divisible(256*self.scale), kernel_size=1, stride=1, padding=0, bias=False)


        self.deconv_layers1 = self._make_deconv_layer(
            1,
            [make_divisible(256*self.scale)//8*8],
            [4],
        )
        self.deconv_layers2 = self._make_deconv_layer(
            1,
            [make_divisible(256*self.scale)],
            [4],
        )
        self.deconv_layers3 = self._make_deconv_layer(
            1,
            [make_divisible(256*self.scale)],
            [4],
        )
        self.deconv_layers4 = self._make_deconv_layer(
            1,
            [make_divisible(256*self.scale)],
            [4],
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                inchannel = make_divisible(256*self.scale)
                fc = nn.Sequential(
                    nn.Conv2d(
                        inchannel,
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0))
            else:
                inchannel = make_divisible(256*self.scale)
                fc = nn.Conv2d(
                    in_channels=inchannel,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.__setattr__(head, fc)

    def init_weights(self, pretrained=True):
        if pretrained:
            for _, m in self.deconv_layers1.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.deconv_layers2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.deconv_layers3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.deconv_layers4.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head or 'ftype' in head or 'cls' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 7:
            padding = 3
            output_padding = 0
        return deconv_kernel, padding, output_padding
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x0 = self.blocks2(x)
        x1 = self.blocks3(x0)
        x2 = self.blocks4(x1)
        x3 = self.blocks5(x2)
        x4 = self.blocks6(x3)

        x3_ = self.deconv_layers1(x4)
        x3_ = self.adaption3(x3) + x3_

        x2_ = self.deconv_layers2(x3_)
        x2_ = self.adaption2(x2) + x2_

        x1_ = self.deconv_layers3(x2_)
        x1_ = self.adaption1(x1) + x1_

        x0_ = self.deconv_layers4(x1_) + self.adaption0(x0)
        x0_ = self.adaptionU1(x0_)

        ret = {}

        for head in self.heads:
            ret[head] = self.__getattr__(head)(x0_)
        return [ret]


def CardDetectionCorrectionModel(ratio = 1., need_ftype=False):
    if need_ftype:
        heads = {'hm': 1, 'cls': 4, 'ftype': 2, 'wh': 8, 'reg': 2}
    else:
        heads = {'hm': 1, 'cls': 4, 'wh': 8, 'reg': 2}
    model = PPLCNet(heads,ratio)
    model.init_weights()
    return model