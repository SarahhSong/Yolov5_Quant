import sys

sys.path.append('../Yolov5_Quant')
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks/python_packages/")


import os
import tqdm
import numpy as np
import random
import tensorflow as tf
import megengine as mge
import megengine.optimizer as optim
from megengine.autodiff import GradManager
from data.visual_ops import draw_bounding_box
from data.generate_coco_data import CoCoDataGenrator
from yolo import Yolo
from loss import ComputeLoss
from val import val
from layers import nms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    epochs = 300
    log_dir = './logs'
    # 可以选择 ['5l', '5s', '5m', '5x']
    yolov5_type = "5s"
    image_shape = (320, 320, 3)
    num_class = 81
    # num_class = 2
    batch_size = 32
    # -1表示全部数据参与训练
    train_img_nums = -1
    train_coco_json = '/content/coco2017/annotations/instances_train2017.json'
    val_coco_json = '/content/coco2017/annotations/instances_val2017.json'
    train_coco_data = '/content/coco2017/train2017'
    val_coco_data = '/content/coco2017/val2017'

    # 类别名, 也可以自己提供一个数组, 不通过coco
    classes = ['none', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none', 'none', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none',
                'dining table', 'none', 'none', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'none', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # classes = ['cat', 'dog']

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[8, 10], [21, 26], [33, 64],
                        [65, 34], [58, 123], [102, 74],
                        [117, 182], [204, 107], [248, 220]]) / 640.
    anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # tensorboard日志
    summary_writer = tf.summary.create_file_writer(log_dir)
    # data generator
    
    coco_data = CoCoDataGenrator(
        coco_annotation_file= train_coco_json,
        coco_data_file = train_coco_data,
        train_img_nums=train_img_nums,
        img_shape=image_shape,
        batch_size=batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=True
    )
    # 验证集
    val_coco_data = CoCoDataGenrator(
        coco_annotation_file=val_coco_json,
        coco_data_file = val_coco_data,
        train_img_nums=-1,
        img_shape=image_shape,
        batch_size=batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=False
    )
    
    yolo = Yolo(
        image_shape=image_shape,
        batch_size=batch_size,
        num_class=num_class,
        is_training=True,
        anchors=anchors,
        anchor_masks=anchor_masks,
        net_type=yolov5_type
    )
    yolo.yolov5.named_modules
    # yolo.yolov5.summary(line_length=200)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    optimizer = optim.Adam(yolo.yolov5.parameters() , lr = 0.0001)

    loss_fn = ComputeLoss(
        image_shape=image_shape,
        anchors=anchors,
        anchor_masks=anchor_masks,
        num_class=num_class,
        anchor_ratio_thres=4,
        only_best_anchor=False,
        balanced_rate=15,
        iou_ignore_thres=0.5
    )

    pre_mAP = 0.
    # data = coco_data.next_batch()
    for epoch in range(epochs):
        train_progress_bar = tqdm.tqdm(range(coco_data.total_batch_size), desc="train epoch {}/{}".format(epoch, epochs-1), ncols=100)
        for batch in train_progress_bar:
            with GradManager() as gm:
                gm.attach(yolo.yolov5.parameters())
                data = coco_data.next_batch()
                valid_nums = data['valid_nums']
                gt_imgs = np.array(data['imgs'] / 255., dtype=np.float32)
                gt_imgs = mge.Tensor(gt_imgs)
                gt_boxes = np.array(data['bboxes'] / image_shape[0], dtype=np.float32)
                gt_classes = data['labels']

                # print("-------epoch {}, step {}, total step {}--------".format(epoch, batch,
                #                                                                epoch * coco_data.total_batch_size + batch))
                # print("current data index: ",
                #       coco_data.img_ids[(coco_data.current_batch_index - 1) * coco_data.batch_size:
                #                         coco_data.current_batch_index * coco_data.batch_size])
                # for i, nums in enumerate(valid_nums):
                #     print("gt boxes: ", gt_boxes[i, :nums, :] * image_shape[0])
                #     print("gt classes: ", gt_classes[i, :nums])
                print(gt_imgs.shape)
                yolo_preds = yolo.yolov5(gt_imgs)
                loss_xy, loss_wh, loss_box, loss_obj, loss_cls = loss_fn(yolo_preds, gt_boxes, gt_classes)

                total_loss = loss_box + loss_obj + loss_cls
                train_progress_bar.set_postfix(ordered_dict={"loss":'{:.5f}'.format(total_loss)})

                # grad = tape.gradient(total_loss, yolo.yolov5.trainable_variables)
                gm.backward(total_loss)
                # optimizer.apply_gradients(zip(grad, yolo.yolov5.trainable_variables))
                optimizer.step().clear_grad()

                # Scalar
                with summary_writer.as_default():
                    tf.summary.scalar('loss/box_loss', loss_box,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/object_loss', loss_obj,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/class_loss', loss_cls,
                                      step=epoch * coco_data.total_batch_size + batch)
                    tf.summary.scalar('loss/total_loss', total_loss,
                                      step=epoch * coco_data.total_batch_size + batch)

                # image, 只拿每个batch的其中一张
                random_one = random.choice(range(batch_size))
                # gt
                gt_img = gt_imgs[random_one].copy() * 255
                gt_box = gt_boxes[random_one] * image_shape[0]
                gt_class = gt_classes[random_one]
                non_zero_ids = np.where(np.sum(gt_box, axis=-1))[0]
                for i in non_zero_ids:
                    cls = gt_class[i]
                    class_name = coco_data.coco.cats[cls]['name']
                    xmin, ymin, xmax, ymax = gt_box[i]
                    # print(xmin, ymin, xmax, ymax)
                    gt_img = draw_bounding_box(gt_img, class_name, cls, int(xmin), int(ymin), int(xmax), int(ymax))

                # pred, 同样只拿第一个batch的pred
                pred_img = gt_imgs[random_one].copy() * 255
                yolo_head_output = yolo.yolo_head(yolo_preds, is_training=False)
                nms_output = nms(image_shape, yolo_head_output.numpy(), iou_thres=0.3)
                if len(nms_output) == batch_size:
                    nms_output = nms_output[random_one]
                    for box_obj_cls in nms_output:
                        if box_obj_cls[4] > 0.5:
                            label = int(box_obj_cls[5])
                            if coco_data.coco.cats.get(label):
                                class_name = coco_data.coco.cats[label]['name']
                                # class_name = classes[label]
                                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                                pred_img = draw_bounding_box(pred_img, class_name, box_obj_cls[4], int(xmin), int(ymin),
                                                             int(xmax), int(ymax))

                concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                summ_imgs = tf.expand_dims(concat_imgs, 0)
                summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                with summary_writer.as_default():
                    tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                                     step=epoch * coco_data.total_batch_size + batch)
        # 这里计算一下训练集的mAP
        val(model=yolo, val_data_generator=coco_data, classes=classes, desc='training dataset val')
        # 这里计算验证集的mAP
        mAP50, mAP, final_df = val(model=yolo, val_data_generator=val_coco_data, classes=classes, desc='val dataset val')
        if mAP > pre_mAP:
            pre_mAP = mAP
            # yolo.yolov5.save_weights(log_dir+"/yolov{}-best.h5".format(yolov5_type))
            mge.save({
                "epoch": epoch,
                "state_dict": yolo.yolov5.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": mAP
               }, './log_dor/yolov{}-best.pkl'.format(yolov5_type))
            print("save {}/yolov{}-best.pkl best weight with {} mAP.".format(log_dir, yolov5_type, mAP))
        # yolo.yolov5.save_weights(log_dir+"/yolov{}-last.h5".format(yolov5_type))
        if epoch % 10 == 0:
          mge.save({
                "epoch": epoch,
                "state_dict": yolo.yolov5.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": mAP
               }, './log_dor/yolov{}-last.pkl'.format(yolov5_type))
          print("save {}/yolov{}-last.pkl last weights at epoch {}.".format(log_dir, yolov5_type, epoch))


if __name__ == "__main__":
    main()
