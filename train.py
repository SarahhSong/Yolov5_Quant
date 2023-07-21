import sys

sys.path.append('../Yolov5_Quant')
sys.path.insert(0,"/content/drive/MyDrive/Colab Notebooks/python_packages/")

import argparse
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
from megengine.quantization import quantize
from yolo import Yolo, convert_qat
from loss import ComputeLoss
from val import val
from layers import nms
from utils.printModel import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging = mge.logger.get_logger()


def main():
    parser = argparse.ArgumentParser(description="MegEngine COCO Q_yolov5s Training")
    # parser.add_argument("-d", "--data", metavar="DIR", help="path to imagenet dataset")
    parser.add_argument(
        "-a",
        "--arch",
        default="5s",
        help="model architecture (default: 5s)",
    )
    parser.add_argument(
        "-m", "--model", metavar="PKL", default=None, help="path to model checkpoint"
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="./logs",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=300,
        type=int,
        help="number of total epochs to run (default: 300)",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="SIZE",
        default=32,
        type=int,
        help="batch size for single GPU (default: 32)",
    )
    parser.add_argument(
        "-n",
        "--img-num",
        default=-1,
        type=int,
        help="image nums for training (default: -1)",
    )
    parser.add_argument(
        "-s",
        "--img-shape",
        default=(320, 320, 3),
        type=tuple,
        help="image shape for model imput" 
    )   
    parser.add_argument(
        "--lr",
        "--learning-rate",
        metavar="LR",
        default=0.0001,
        type=float,
        help="learning rate for single GPU (default: 0.0001)",
    )
    # parser.add_argument(
    #     "--momentum", default=0.9, type=float, help="momentum (default: 0.9)"
    # )
    # parser.add_argument(
    #     "--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)"
    # )
    parser.add_argument(
        "--mode",
        default="normal",
        type=str,
        choices=["normal", "qat"],
        help="Quantization Mode\n"
        "normal: no quantization, using float32\n"
        "qat: quantization aware training, simulate int8\n"
    )
    parser.add_argument(
        "--train-json",
        default='/mnt/e/数据集/COCO/annotations/annotations_trainval2017/annotations/instances_train2017.json',
        type=str,
        help="dir for train_data_json (default: )" 
    )       
    parser.add_argument(
        "--val-json",
        default='/mnt/e/数据集/COCO/annotations/annotations_trainval2017/annotations/instances_val2017.json',
        type=str,
        help="dir for val_data_json (default: )" 
    )
    parser.add_argument(
        "--train-data",
        default='/mnt/e/数据集/COCO/images/train2017/train2017',
        type=str,
        help="dir for train_data (default: )" 
    )
    parser.add_argument(
        "--val-data",
        default='/mnt/e/数据集/COCO/images/val2017/val2017',
        type=str,
        help="dir for val_data (default:)" 
    )             
    args = parser.parse_args()
    worker(args)
    

def worker(args):
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
    num_class = len(classes)

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[8, 10], [21, 26], [33, 64],
                        [65, 34], [58, 123], [102, 74],
                        [117, 182], [204, 107], [248, 220]]) / 640.
    anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # tensorboard日志
    # data generator
    
    yolo = Yolo(
        image_shape=args.img_shape,
        batch_size=args.batch_size,
        num_class=num_class,
        is_training=True,
        anchors=anchors,
        anchor_masks=anchor_masks,
        net_type=args.arch
    )

    model = yolo.yolov5

    if args.model is not None:
        logging.info("load from checkpoint %s", args.model)
        checkpoint = mge.load(args.model)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    if args.mode == "qat":
        model = convert_qat(model)

    # if args.mode == "quantize":
    #     model = quantize(model) 
    #     with open('model.txt', 'w') as f:
    #         for k, v in model.named_parameters():
    #             f.write("{}\t{}\n".format(k, v))

    optimizer = optim.Adam(model.parameters() , args.lr)

    loss_fn = ComputeLoss(
        image_shape=args.img_shape,
        anchors=anchors,
        anchor_masks=anchor_masks,
        num_class=num_class,
        anchor_ratio_thres=4,
        only_best_anchor=False,
        balanced_rate=15,
        iou_ignore_thres=0.5
    )

    pre_mAP = 0.

    coco_data, val_coco_data = build_datasets(args)
    summary_writer = tf.summary.create_file_writer(args.save)

    for epoch in range(args.epochs):
        train_progress_bar = tqdm.tqdm(range(coco_data.total_batch_size), desc="train epoch {}/{}".format(epoch, args.epochs-1), ncols=100)
        for batch in train_progress_bar:
            with GradManager() as gm:
                gm.attach(model.parameters())
                data = coco_data.next_batch()
                valid_nums = data['valid_nums']
                gt_imgs = np.array(data['imgs'] / 255., dtype=np.float32)
                gt_imgs = mge.Tensor(gt_imgs)
                gt_boxes = np.array(data['bboxes'] / args.img_shape[0], dtype=np.float32)
                gt_classes = data['labels']

                # print("-------epoch {}, step {}, total step {}--------".format(epoch, batch,
                #                                                                epoch * coco_data.total_batch_size + batch))
                # print("current data index: ",
                #       coco_data.img_ids[(coco_data.current_batch_index - 1) * coco_data.batch_size:
                #                         coco_data.current_batch_index * coco_data.batch_size])
                # for i, nums in enumerate(valid_nums):
                #     print("gt boxes: ", gt_boxes[i, :nums, :] * image_shape[0])
                #     print("gt classes: ", gt_classes[i, :nums])
                yolo_preds = model(gt_imgs)
                loss_xy, loss_wh, loss_box, loss_obj, loss_cls = loss_fn(yolo_preds, gt_boxes, gt_classes)
                # print("loss_box:{}, loss_cls:{}, loss_obj:{}".format(loss_box, loss_cls, loss_obj))

                total_loss = loss_box + loss_obj + loss_cls
                train_progress_bar.set_postfix(ordered_dict={"loss":'{:.5f}'.format(total_loss)})

                gm.backward(mge.Tensor(np.array(total_loss)))

                with open('grad.txt', 'w') as f:
                    for name, paramer in model.named_parameters():
                        f.write("-->name:{}\t-->grad_value:{}\n".format(name, paramer.grad))

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

                # # image, 只拿每个batch的其中一张
                random_one = random.choice(range(args.batch_size))
                # gt
                gt_imgs = gt_imgs.numpy()
                gt_img = gt_imgs[random_one].copy() * 255
                gt_box = gt_boxes[random_one] * args.img_shape[0]
                gt_class = gt_classes[random_one]
                non_zero_ids = np.where(np.sum(gt_box, axis=-1))[0]
                for i in non_zero_ids:
                    cls = gt_class[i]
                    class_name = coco_data.coco.cats[cls]['name']
                    xmin, ymin, xmax, ymax = gt_box[i]
                    # print(xmin, ymin, xmax, ymax)
                    gt_img = draw_bounding_box(gt_img, class_name, cls, int(xmin), int(ymin), int(xmax), int(ymax))

                # # pred, 同样只拿第一个batch的pred
                pred_img = gt_imgs[random_one].copy() * 255
                yolo_head_output = yolo.yolo_head(yolo_preds, is_training=False)
                nms_output = nms(args.img_shape, yolo_head_output.numpy(), iou_thres=0.3)
                if len(nms_output) == args.batch_size:
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

                # concat_imgs = tf.concat([gt_img[:, :, ::-1], pred_img[:, :, ::-1]], axis=1)
                # summ_imgs = tf.expand_dims(concat_imgs, 0)
                # summ_imgs = tf.cast(summ_imgs, dtype=tf.uint8)
                # with summary_writer.as_default():
                #     tf.summary.image("imgs/gt,pred,epoch{}".format(epoch), summ_imgs,
                #                      step=epoch * coco_data.total_batch_size + batch)
        # 这里计算一下训练集的mAP
        val(model=yolo, val_data_generator=coco_data, classes=classes, desc='training dataset val')
        # 这里计算验证集的mAP
        mAP50, mAP, final_df = val(model=yolo, val_data_generator=val_coco_data, classes=classes, desc='val dataset val')
        if mAP > pre_mAP:
            pre_mAP = mAP
            # yolo.yolov5.save_weights(log_dir+"/yolov{}-best.h5".format(yolov5_type))
            mge.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": mAP
                }, './logs/yolov{}-best.pkl'.format(args.arch))
            print("save {}/yolov{}-best.pkl best weight with {} mAP.".format(args.save, args.arch, mAP))
        # yolo.yolov5.save_weights(log_dir+"/yolov{}-last.h5".format(yolov5_type))

        if epoch % 10 == 0:
          mge.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mAP": mAP
                }, './logs/yolov{}-epoch{}.pkl'.format(args.arch, epoch))
          print("save {}/yolov{}-epcho.pkl last weights at epoch {}.".format(args.save, args.arch, epoch))


def build_datasets(args):
    # 训练集
    coco_data = CoCoDataGenrator(
        coco_annotation_file= args.train_json,
        coco_data_file = args.train_data,
        train_img_nums=args.img_num,
        img_shape=args.img_shape,
        batch_size=args.batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=True
    )
    # 验证集
    val_coco_data = CoCoDataGenrator(
        coco_annotation_file=args.val_json,
        coco_data_file = args.val_data,
        train_img_nums=args.img_num,
        img_shape=args.img_shape,
        batch_size=args.batch_size,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False,
        need_down_image=False,
        using_argument=False
    )
    return coco_data, val_coco_data


if __name__ == "__main__":
    main()
