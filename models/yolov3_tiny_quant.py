import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
import torch.nn as nn
import torch


class QuantSiLU(QuantNLAL):
    def __init__(
            self,
            act_quant=Uint8ActPerTensorFloat,
            input_quant=None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.SiLU,
            passthrough_act=False,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, weight_bit_width=8, bias_quant=Int32Bias,
                      output_quant=Int8ActPerTensorFloat, bias=True, activation=QuantSiLU(return_quant_tensor=True)):
    layers = [qnn.QuantConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias,
                              weight_bit_width=weight_bit_width, bias_quant=bias_quant, output_quant=output_quant,
                              return_quant_tensor=True)]
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)


class Yolov3TinyQuant(nn.Module):
    def __init__(self, n_classes, n_anchors, weight_bit_width=8, bias_quant=Int32Bias,
                 output_quant=Int8ActPerTensorFloat):
        super(Yolov3TinyQuant, self).__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.in_quant = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.backbone = nn.ModuleList()
        self.backbone.append(create_conv_block(3, 16, 3, 1, 1,
                                               weight_bit_width, bias_quant, output_quant))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(16, 32, 3, 1, 1,
                                               weight_bit_width, bias_quant, output_quant))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(32, 64, 3, 1, 1,
                                               weight_bit_width, bias_quant, output_quant))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(64, 128, 3, 1, 1,
                                               weight_bit_width, bias_quant, output_quant))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(128, 256, 3, 1, 1,
                                               weight_bit_width, bias_quant, output_quant))

        self.big_obj_head = nn.ModuleList()
        self.big_obj_head.append(qnn.QuantMaxPool2d(2, 2, return_quant_tensor=True))
        self.big_obj_head.append(create_conv_block(256, 512, 3, 1, 1,
                                                   weight_bit_width, bias_quant, output_quant))
        self.big_obj_head.append(nn.ZeroPad2d((0, 1, 0, 1)))
        self.big_obj_head.append(qnn.QuantMaxPool2d(2, 1, return_quant_tensor=True))
        self.big_obj_head.append(create_conv_block(512, 1024, 3, 1, 1,
                                                   weight_bit_width, bias_quant, output_quant))
        self.big_obj_head.append(create_conv_block(1024, 256, 1, 1, 0,
                                                   weight_bit_width, bias_quant, output_quant))
        self.big_obj_head.append(create_conv_block(256, 512, 3, 1, 1,
                                                   weight_bit_width, bias_quant, output_quant))

        self.small_obj_head = nn.ModuleList()
        self.small_obj_head.append(create_conv_block(256, 128, 1, 1, 0,
                                                     weight_bit_width, bias_quant, output_quant))
        self.small_obj_head.append(qnn.QuantUpsample(scale_factor=2, return_quant_tensor=True))
        self.small_obj_head.append(create_conv_block(384, 256, 3, 1, 1,
                                                     weight_bit_width, bias_quant, output_quant))

        self.big_obj_out = qnn.QuantConv2d(512, n_anchors * (5 + n_classes), 1, 1, 0,
                                           weight_bit_width=weight_bit_width, bias_quant=bias_quant,
                                           output_quant=output_quant)
        self.small_obj_out = qnn.QuantConv2d(256, n_anchors * (5 + n_classes), 1, 1, 0,
                                             weight_bit_width=weight_bit_width, bias_quant=bias_quant,
                                             output_quant=output_quant)

    def forward(self, x):
        backbone_output = self.in_quant(x)
        for layer in self.backbone:
             backbone_output = layer(backbone_output)

        big_head_output_0 = backbone_output
        for layer in self.big_obj_head[:6]:
            big_head_output_0 = layer(big_head_output_0)

        big_head_output_1 = big_head_output_0
        for layer in self.big_obj_head[6:]:
            big_head_output_1 = layer(big_head_output_1)

        small_head_output = big_head_output_0
        for layer in self.small_obj_head[:2]:
            small_head_output = layer(small_head_output)

        small_head_output = torch.cat([small_head_output, backbone_output], dim=1)
        for layer in self.small_obj_head[2:]:
            small_head_output = layer(small_head_output)

        return [self.small_obj_out(small_head_output), self.big_obj_out(big_head_output_1)]


