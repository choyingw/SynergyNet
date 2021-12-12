import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=self.padding, 
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride)
        return x

class DCNv2(nn.Module):
    def __init__(self,
                num_classes=1000,
                width_mult=1.0,
                inverted_residual_setting=None,
                round_nearest=8,
                block=None,
                norm_layer=None):

        super(DCNv2, self).__init__()

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, deform
                (1, 16, 1, 1, False),
                (6, 24, 2, 2, False),
                (6, 32, 3, 2, False),
                (6, 64, 4, 2, True),
                (6, 96, 3, 1, True),
                (6, 160, 3, 2, True),
                (6, 320, 1, 1, False),
            ]

        input_channel = 32
        input_channel = int(input_channel * width_mult)

        features = [nn.Conv2d(3, input_channel, kernel_size=1, stride=2)]

        for t, c, n, s, deformable in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                conv = nn.Conv2d # if deformable==False else DeformableConv2d
                print('{} {}', input_channel, output_channel)
                features.append(conv(input_channel, output_channel, kernel_size=1, stride=stride, bias=True))
                input_channel = output_channel

        last_channel = 1280
        self.last_channel = int(last_channel * max(1.0, width_mult))

        features.append(nn.Conv2d(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10

        self.classifier_ori = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_ori),
        )
        self.classifier_shape = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_shape),
        )
        self.classifier_exp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_exp),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        x = self.features(x)

        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1)

        pool_x = x.clone()

        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)

        x = torch.cat((x_ori, x_shape, x_exp), dim=1)
        return x, pool_x

    def forward(self, x):
        return self._forward_impl(x)

def dcnv2(pretrained=False, progress=True, **kwargs):
    model = DCNv2(**kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict, strict=False)
    return model