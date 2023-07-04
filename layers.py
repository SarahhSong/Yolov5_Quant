import megengine.functional as F
import megengine.module as M
import megengine as mge
from collections import OrderedDict


class UpSample(M.Module):

    def __init__(self, scale_factor=2, mode="bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.vision.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class SiLU(M.Module):
    """export-friendly version of M.SiLU()"""

    @staticmethod
    def forward(x):
        return x * F.sigmoid(x)


def get_activation(name="silu"):
    if name == "silu":
        if hasattr(M, "SiLU"):
            module = M.SiLU()
        else:
            module = SiLU()
    elif name == "relu":
        module = M.ReLU()
    elif name == "lrelu":
        module = M.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(M.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = M.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = M.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(M.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1,
            stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(M.Module):
    # Standard bottleneck
    def __init__(
        self, in_channels, out_channels, shortcut=True,
        expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(M.Module):
    "Residual layer with `in_channels` inputs."
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


# class SPPBottleneck(M.Module):
#     """Spatial pyramid pooling layer used in YOLOv3-SPP"""
#     def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
#         super().__init__()
#         hidden_channels = in_channels // 2
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
#         self.m = [M.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
#         conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
#         self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.concat([x] + [m(x) for m in self.m], axis=1)
#         x = self.conv2(x)
#         return x


class C3(M.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels, out_channels, n=1,
        shortcut=True, expansion=0.5, depthwise=False, act="silu"
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = M.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = F.concat((x_1, x_2), axis=1)
        return self.conv3(x)


class Focus(M.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = F.concat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), axis=1,
        )
        return self.conv(x)


class SPPF(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        in_half_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, in_half_channels, 1, 1)
        self.conv2 = BaseConv(in_half_channels*4, out_channels, 1, 1)
        self.maxpool = M.MaxPool2d(kernel_size=kernel_size, strides=1, padding=kernel_size//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        concat_all = F.concat([x, y1, y2, y3],axis=1)
        output = self.conv2(concat_all)
        return output

class Concat(M.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return F.concat(x,self.d)

def meshgrid(x, y):
    # meshgrid wrapper for megengine
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y 
 
class detect(M.Module):
    stride = None
    onnx_dynamic = False  # ONNX export parameter
    
    def __init__(self, nc = 80, anchors = (), ch =(), inplace = True):
        super().__init__()
        self.nc = nc
        self.no = nc+5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [F.zeros((1))] * self.nl
        self.anchor_grid = [F.zeros((1))] * self.nl
        self.inplace = inplace
        modules = OrderedDict()
        for idx, x in enumerate(ch):
            modules[idx] = M.Conv2d(x, self.no * self.na, 1)
        self.m = M.Sequential(modules)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx)
            x[i] = F.transpose(x[i], (0, 1, 3, 4, 2))
        if not self.training:
            if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
            
            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] *2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] *2) ** 2 * self.anchor_grid[i]
            else:
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                y = F.concat([xy, wh, y[..., 4:]], -1)
            y = y.reshape(bs, -1, self.no)
            z.append(y)

        return x if self.training else (F.concat(z,axis=1), x) 
            

    def _make_grid(self, nx = 20, ny = 20, i = 0):
        d = self.anchors[i].device
        yv, xv = meshgrid(F.arange(ny), F.arange(nx))
        grid = F.stack((xv, yv), axis=2).expand_dims(0).broadcast_to((1, self.na, ny, nx, 2)).astype("float32")
        t = self.anchors[i]
        anchor_grid = (t * self.stride[i]).reshape((1, self.na, 1, 1, 2)).broadcast_to((1, self.na, ny, nx, 2)).astype("float32")
        return grid, anchor_grid

