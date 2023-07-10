import megengine.functional as F
import megengine.module as M
import megengine as mge
from collections import OrderedDict
import numpy as np

class Upsample(M.Module):

    def __init__(self, scale_factor=2., mode="nearest"):
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


class Conv(M.Module):
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
        self.dconv = Conv(
            in_channels, in_channels, ksize=ksize,
            stride=stride, groups=in_channels, act=act
        )
        self.pconv = Conv(
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
        # Conv = DWConv if depthwise else Conv
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


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
        self.conv1 = Conv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = Conv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
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


class SPPF(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        in_half_channels = in_channels // 2
        self.conv1 = Conv(in_channels, in_half_channels, 1, 1)
        self.conv2 = Conv(in_half_channels*4, out_channels, 1, 1)
        self.maxpool = M.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
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
        for i, t in enumerate(x):
            # print(type(t))
            if isinstance(t, tuple):
                x[i] = t[0]
                # print(type(x[i]))
        # <class 'tuple'>
        # <class 'megengine.tensor.Tensor'>
        return F.concat(x, axis=self.d)


class Reshape(M.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.t_shape = target_shape
    
    def forward(self, x):
        return F.reshape(x, self.t_shape)

# tested
def meshgrid(x, y):
    # meshgrid wrapper for megengine
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    mesh_shape = (y.shape[0], x.shape[0])
    mesh_x = F.broadcast_to(x, mesh_shape)
    mesh_y = F.broadcast_to(y.reshape(-1, 1), mesh_shape)
    return mesh_x, mesh_y 
 

class YoloHead(M.Module):
    def __init__(self, image_shape, num_class, is_training, strides, anchors, anchors_masks):
        super().__init__()
        self.image_shape = image_shape
        self.num_class = num_class
        self.is_training = is_training
        self.strides = strides
        self.anchors = anchors
        self.anchors_masks = anchors_masks
        self.grid = []
        self.anchor_grid = []
        for i, stride in enumerate(strides):
            grid, anchor_grid = self._make_grid(self.image_shape[0] // stride, self.image_shape[1] // stride, i)
            self.grid.append(grid)
            self.anchor_grid.append(anchor_grid)

    def forward(self, inputs):
        detect_res = []
        for i, pred in enumerate(inputs):
            if not self.is_training:
                pred = F.sigmoid(pred)
                f_shape = pred.shape
                # if len(self.grid) < self.anchor_masks.shape[0]:
                #     grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                #     self.grid.append(grid)
                #     self.anchor_grid.append(anchor_grid)
                # 这里把输出的值域从[0,1]调整到[0, image_shape]
                # pred_xy = (tf.sigmoid(pred[..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.strides[i]
                pred_xy = (pred[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.strides[i]
                # pred_wh = (tf.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                pred_wh = (pred[..., 2:4] * 2) * (pred[..., 2:4] * 2) * self.anchor_grid[i]
                # print(self.grid)
                pred_obj = pred[..., 4:5]
                # pred_cls = tf.keras.layers.Softmax()(pred[..., 5:])
                pred_cls = pred[..., 5:]
                cur_layer_pred_res = M.Concat([pred_xy, pred_wh, pred_obj, pred_cls], axis=-1)

                # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
                cur_layer_pred_res = Reshape([f_shape[1]*f_shape[2]*f_shape[3], self.num_class + 5])(cur_layer_pred_res)
                detect_res.append(cur_layer_pred_res)
            else:
                detect_res.append(pred)
        return detect_res if self.is_training else F.concat(detect_res, axis=1)
    
    # tested
    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchors_masks[i]] * np.array([[self.image_shape[1], self.image_shape[0]]])
        cur_layer_anchors = mge.Tensor(cur_layer_anchors)
        num_anchors_per_layer = len(cur_layer_anchors)
        yv, xv = meshgrid(F.arange(h), F.arange(w))
        grid = F.stack((xv, yv), axis=2)
        # 用来计算中心点的grid cell左上角坐标
        grid = F.tile(F.reshape(grid, [1, h, w, 1, 2]), [1, 1, 1, num_anchors_per_layer, 1])
        grid = mge.Tensor(grid, dtype = "float32")
        # anchor_grid = tf.reshape(cur_layer_anchors * self.strides[i], [1, 1, 1, num_anchors_per_layer, 2])
        anchor_grid = F.reshape(cur_layer_anchors, [1, 1, 1, num_anchors_per_layer, 2])
        # 用来计算宽高的anchor w/h
        anchor_grid = F.tile(anchor_grid, [1, h, w, 1, 1])
        anchor_grid = mge.Tensor(anchor_grid, dtype = "float32")

        return grid, anchor_grid

def nms(image_shape, predicts, conf_thres=0.45, iou_thres=0.2, max_det=300, max_nms=3000):
    
    output = []

    for i, predict in enumerate(predicts):
        obj_mask = predict[..., 4] > conf_thres
        predict = predict[obj_mask]

        if not predict.shape[0]:
            continue
        predict[:, 5:] *= predict[:, 4:5]

        x1 = np.maximum(predict[:, 0] - predict[:, 2] / 2, 0)
        y1 = np.maximum(predict[:, 1] - predict[:, 3] / 2, 0)
        x2 = np.minimum(predict[:, 0] + predict[:, 2] / 2, image_shape[1])
        y2 = np.minimum(predict[:, 1] + predict[:, 3] / 2, image_shape[0])
        box = np.concatenate([x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=-1)
        # Detections matrix [n, (x1, y1, x2, y2, conf, cls)]
        max_cls_ids = np.array(predict[:, 5:].argmax(axis=1), dtype=np.float32)
        max_cls_score = predict[:, 5:].max(axis=1)
        predict = np.concatenate([box, max_cls_score[:, None], max_cls_ids[:, None]], axis=1)[
            np.reshape(max_cls_score > 0.1, (-1,))]

        n = predict.shape[0]
        if not n:
            continue
        elif n > max_nms:
            predict = predict[predict[:, 4].argsort()[::-1][:max_nms]]

        # 为每个类别乘上一个大数,box再加上这个偏移, 做nms时就可以在类内做
        cls = predict[:, 5:6] * 4096
        # 边框加偏移
        boxes, scores = predict[:, :4] + cls, predict[:, 4]
        nms_ids = F.vision.nms(
            boxes, scores, iou_thres, max_det)

        output.append(predict[nms_ids.numpy()])

    return output

if __name__ == "__main__":
    # c3 = C3(64,64)
    # print(c3)
    # sppf = SPPF(512,1024)
    # print(sppf) # c3 = C3(64,64)

    x = F.ones(32 * 320 * 320 * 3)
    x = F.reshape(x , [32,320,320,3])
    print(x.shape)
    conv1 = Conv(3,32,6,2)
    print(conv1(x))


