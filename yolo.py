import sys

sys.path.append("../Yolov5_Quant")

import os
import numpy as np
import cv2
import megengine as mge
import megengine.functional as F
import megengine.module as M
from yolov5s import Yolov5s
from layers import nms, meshgrid, YoloHead, Reshape, Concat
from utils.printModel import summary
import megengine.module.qat as qat
from megengine.quantization.qconfig import QConfig
from megengine.quantization.quantize import quantize, quantize_qat
from functools import partial
import megengine.quantization as Q
from megengine.quantization import fake_quant
from megengine.core.tensor.dtype import QuantDtypeMeta
import math
from typing import Union

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Yolov5(M.Module):
        def __init__(self,
            image_shape,
            batch_size,
            num_class,
            anchors_per_location,
            is_training,
            strides,
            anchors,
            anchors_masks):
            super().__init__()
            self.backbone = Yolov5s(image_shape,
                        batch_size=batch_size,
                        num_class=num_class,
                        anchors_per_location=anchors_per_location,)
            self.head = YoloHead(image_shape=image_shape,
                    num_class=num_class,
                    batch_size=batch_size,
                    is_training=is_training,
                    strides=strides,
                    anchors=anchors,
                    anchors_masks=anchors_masks)
            self.quant = M.QuantStub()
            self.dequant = M.DequantStub()
        def forward(self, x):
            x = self.quant(x)
            x = self.backbone(x)            
            x = self.dequant(x)
            x = self.head(x)
            return x  # 要有返回值啊！     

class Yolo:

    def __init__(self,
                 num_class,
                 anchors,
                 anchor_masks,
                 image_shape=[640, 640, 3],
                 is_training=True,
                 batch_size=5,
                 net_type='5s',
                 strides=[8, 16, 32],
                 anchors_per_location=3,
                 yolo_max_boxes=100,
                 yolo_iou_threshold=0.3,
                 yolo_conf_threshold=0.5,
                 model_path=None):
        self.image_shape = image_shape
        self.is_training = is_training
        self.batch_size = batch_size
        self.net_type = net_type
        self.strides = strides
        self.anchors_per_location = anchors_per_location
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_conf_threshold = yolo_conf_threshold
        self.model_path = model_path

        self.num_class = num_class
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.backbone = None

        self.grid = []
        self.anchor_grid = []
        self.yolov5 = Yolov5(image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location,
                is_training=self.is_training,
                strides=self.strides,
                anchors=self.anchors,
                anchors_masks=self.anchor_masks)

    
        if not is_training:
            assert model_path, "Inference mode need the model_path!"
            assert os.path.isfile(model_path), "Can't find the model weight file!"
            checkpoint = mge.load(model_path)
            self.yolov5.load_state_dict(checkpoint["model_state_dict"])

            print("loading model weight from {}".format(model_path))


    def yolo_head(self, features, is_training):
        """ yolo最后输出层wo
        :param features:
        :return: train mode: [[batch, h, w, num_anchors_per_layer, num_class + 5], [...], [...]]
                 infer mode: [batch, -1, num_class + 5]
        """
        # num_anchors_per_layer = len(self.anchors[0])
        detect_res = []
        for i, pred in enumerate(features):
            if not is_training:
                f_shape = pred.shape
                if len(self.grid) < self.anchor_masks.shape[0]:
                    grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                    self.grid.append(grid)
                    self.anchor_grid.append(anchor_grid)
                # 这里把输出的值域从[0,1]调整到[0, image_shape]
                pred_xy = (F.sigmoid(pred[..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.strides[i]
                pred_wh = (F.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                # print(self.grid)
                pred_obj = F.sigmoid(pred[..., 4:5])
                pred_cls = F.softmax(pred[..., 5:])
                cur_layer_pred_res = F.concat([pred_xy, pred_wh, pred_obj, pred_cls], -1)

                # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
                cur_layer_pred_res = Reshape([self.batch_size, -1, self.num_class + 5])(cur_layer_pred_res)
                detect_res.append(cur_layer_pred_res)
            else:
                detect_res.append(pred)
        return detect_res if is_training else F.concat(detect_res, axis=1)

    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchor_masks[i]] * np.array([[self.image_shape[1], self.image_shape[0]]])
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
    

    def predict(self, images, image_need_resize=True, resize_to_origin=True):
        """预测
           预测模式下实例化类: is_training=False, weights_path=, batch_size跟随输入建议1, image_shape跟随训练模式,不做调整
        :param images: [batch, h, w, c] or [h, w, c]
        :return [[nms_nums, (x1, y1, x2, y2, conf, cls)], [...], [...], ...]
        """

        if len(np.shape(images)) <= 3:
            images = [images]
            self.batch_size = 1

        final_outputs = []
        for i, im in enumerate(images):
            if image_need_resize:
                im_shape = np.shape(im)
                im_size_max = np.max(im_shape[0:2])
                im_scale = float(self.image_shape[0]) / float(im_size_max)

                # resize原始图片
                im_resize = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
                im_resize_shape = np.shape(im_resize)
                im_blob = np.zeros(self.image_shape, dtype=np.float32)
                im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize
                inputs = np.array([im_blob], dtype=np.float32) / 255.
            else:
                inputs = np.array([im], dtype=np.float32) / 255.

            outputs = self.yolov5(inputs)

            nms_outputs = nms(self.image_shape, outputs)

            if not nms_outputs:
                continue
            nms_outputs = np.array(nms_outputs[0], dtype=np.float32)

            # resize回原图大小
            if resize_to_origin:
                boxes = nms_outputs[:, :4]
                b0 = np.maximum(np.minimum(boxes[:, 0] / im_scale, im_shape[1] - 1), 0)
                b1 = np.maximum(np.minimum(boxes[:, 1] / im_scale, im_shape[0] - 1), 0)
                b2 = np.maximum(np.minimum(boxes[:, 2] / im_scale, im_shape[1] - 1), 0)
                b3 = np.maximum(np.minimum(boxes[:, 3] / im_scale, im_shape[0] - 1), 0)
                origin_boxes = np.stack([b0, b1, b2, b3], axis=1)
                nms_outputs[:, :4] = origin_boxes

            final_outputs.append(nms_outputs)
        final_outputs = np.array(final_outputs)

        return final_outputs

class TQT(fake_quant.TQT):
    r"""
    TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.

    :param dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
        quantization dtype of input.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def __init__(
        self, dtype: Union[str, QuantDtypeMeta], enable: bool = True, **kwargs
    ):
        super().__init__(dtype, enable, **kwargs)
        self.scale = mge.Parameter(1.0, dtype="float32")
        self.zero_point = mge.Tensor(0.0, dtype="float32")

    def get_qparams(self):
        return fake_quant.create_qparams(
            fake_quant.QuantMode.SYMMERTIC,
            self.dtype,
            scale=2 ** self.scale,
            zero_point=self.zero_point,
        )


class Int8ToUint4(M.Module):
    def __init__(self):
        super().__init__()
        self.dequant = M.DequantStub()
        self.quant = M.QuantStub()

    def forward(self, x):
        return self.quant(self.dequant(x))


def  convert_qat(model):
    qconfig = QConfig(
        weight_observer=None,
        act_observer=None,
        weight_fake_quant=partial(TQT,dtype="qint4"),
        act_fake_quant=partial(TQT, dtype="quint4")
    )
    for module in model.modules():
        if isinstance(module, M.QuantStub):
            module.disable_quantize()

    model_qat = quantize_qat(model, qconfig=qconfig)
    return model_qat


def convert_quantized(model, model_path):
    qconfig = QConfig(
        weight_observer=None,
        act_observer=None,
        weight_fake_quant=partial(TQT, dtype="qint4"),
        act_fake_quant=partial(TQT, dtype="quint4"),
    )

    model_qat = quantize_qat(model, qconfig=qconfig)

    checkpoint = mge.load(model_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_qat.load_state_dict(state_dict, strict=False)

    # model_qat.int8_to_uint4.quant.act_fake_quant.scale[...] = model_qat.conv_bn_relu1.act_fake_quant.scale.numpy()

    model_quantized = quantize(model_qat)
    return model_quantized


if __name__ == "__main__":
    batch_size = 3
    image_shape = (640, 640, 3)
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / image_shape[0]
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    anchors = np.array(anchors, dtype=np.float32)
    yolo = Yolov5(image_shape, batch_size=batch_size, num_class=80, anchors_per_location=3, strides=[8, 16, 32], is_training=False, anchors=anchors, anchors_masks=anchor_masks)
    imgs = F.arange(batch_size * 3 * 640 * 640)
    imgs = F.reshape(imgs, (batch_size,3,640,640))
    for i in yolo(imgs):
        print(i.shape)

    # yolo.yolov5.summary(line_length=200)
    #s
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    #
    # tb_writer = tf.summary.create_file_writer('./logs')
    # with tb_writer.as_default():
    #     if not yolo3.yolo_model.run_eagerly:
    #         summary_ops_v2.graph(get_graph(), step=0)
