import torch
from torch import nn


def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, batch_norm=False,
                      activation=nn.SiLU(inplace=True), bias=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(activation)
    return nn.Sequential(*layers)


class DetectHead(nn.Module):
    def __init__(self, n_classes, anchors, in_channels):
        super(DetectHead, self).__init__()
        self.nc = n_classes
        self.no = 5 + n_classes
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.stride = torch.tensor([16., 32.]).float().to('cuda:0')

    def forward(self, x):
        out = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            out.append(y.view(bs, self.na * nx * ny, self.no))
        return torch.cat(out, 1)

    def _make_grid(self, nx=20, ny=20, i=0):
        """Generates a grid and corresponding anchor grid with shape `(1, num_anchors, ny, nx, 2)` for indexing
        anchors.
        """
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Yolov3Tiny(nn.Module):
    def __init__(self, n_classes, n_anchors):
        super(Yolov3Tiny, self).__init__()
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.backbone = nn.ModuleList()
        self.backbone.append(create_conv_block(3, 16, 3, 1, 1))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(16, 32, 3, 1, 1))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(32, 64, 3, 1, 1))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(64, 128, 3, 1, 1))
        self.backbone.append(nn.MaxPool2d(2, 2))
        self.backbone.append(create_conv_block(128, 256, 3, 1, 1))

        self.big_obj_head = nn.ModuleList()
        self.big_obj_head.append(nn.MaxPool2d(2, 2))
        self.big_obj_head.append(create_conv_block(256, 512, 3, 1, 1))
        self.big_obj_head.append(nn.ZeroPad2d((0, 1, 0, 1)))
        self.big_obj_head.append(nn.MaxPool2d(2, 1))
        self.big_obj_head.append(create_conv_block(512, 1024, 3, 1, 1))
        self.big_obj_head.append(create_conv_block(1024, 256, 1, 1, 0))
        self.big_obj_head.append(create_conv_block(256, 512, 3, 1, 1))

        self.small_obj_head = nn.ModuleList()
        self.small_obj_head.append(create_conv_block(256, 128, 1, 1, 0))
        self.small_obj_head.append(nn.Upsample(scale_factor=2))
        self.small_obj_head.append(create_conv_block(384, 256, 3, 1, 1))

        self.big_obj_out = nn.Conv2d(512, n_anchors * (5 + n_classes), 1, 1, 0)
        self.small_obj_out = nn.Conv2d(256, n_anchors * (5 + n_classes), 1, 1, 0)

    def forward(self, x):
        backbone_output = x
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


def load_from_ultralytics_state_dict(model, path):
    state_dict = torch.load(path)
    model.backbone[0][0].weight.data = state_dict['model.model.0.conv.weight']    # Conv2d
    model.backbone[0][0].bias.data = state_dict['model.model.0.conv.bias']

    model.backbone[2][0].weight.data = state_dict['model.model.2.conv.weight']
    model.backbone[2][0].bias.data = state_dict['model.model.2.conv.bias']

    model.backbone[4][0].weight.data = state_dict['model.model.4.conv.weight']
    model.backbone[4][0].bias.data = state_dict['model.model.4.conv.bias']

    model.backbone[6][0].weight.data = state_dict['model.model.6.conv.weight']
    model.backbone[6][0].bias.data = state_dict['model.model.6.conv.bias']

    model.backbone[8][0].weight.data = state_dict['model.model.8.conv.weight']
    model.backbone[8][0].bias.data = state_dict['model.model.8.conv.bias']

    model.big_obj_head[1][0].weight.data = state_dict['model.model.10.conv.weight']
    model.big_obj_head[1][0].bias.data = state_dict['model.model.10.conv.bias']

    model.big_obj_head[4][0].weight.data = state_dict['model.model.13.conv.weight']
    model.big_obj_head[4][0].bias.data = state_dict['model.model.13.conv.bias']

    model.big_obj_head[5][0].weight.data = state_dict['model.model.14.conv.weight']
    model.big_obj_head[5][0].bias.data = state_dict['model.model.14.conv.bias']

    model.big_obj_head[6][0].weight.data = state_dict['model.model.15.conv.weight']
    model.big_obj_head[6][0].bias.data = state_dict['model.model.15.conv.bias']

    model.small_obj_head[0][0].weight.data = state_dict['model.model.16.conv.weight']
    model.small_obj_head[0][0].bias.data = state_dict['model.model.16.conv.bias']

    model.small_obj_head[2][0].weight.data = state_dict['model.model.19.conv.weight']
    model.small_obj_head[2][0].bias.data = state_dict['model.model.19.conv.bias']

    model.small_obj_out.weight.data = state_dict['model.model.20.m.0.weight']
    model.small_obj_out.bias.data = state_dict['model.model.20.m.0.bias']

    model.big_obj_out.weight.data = state_dict['model.model.20.m.1.weight']
    model.big_obj_out.bias.data = state_dict['model.model.20.m.1.bias']

