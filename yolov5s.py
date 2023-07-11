import sys

sys.path.append("../Yolov5_Quant")

import megengine.functional as F
import megengine.module as M
from layers import Conv, C3, SPPF, Concat, Upsample, Reshape


class Yolov5s(M.Module):
    def __init__(self, image_shape, batch_size, num_class, anchors_per_location):
        super().__init__()
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.num_class = num_class
        self.anchors_per_location = anchors_per_location

        self.conv1 = Conv(3, 32, ksize=6, stride=2)
        self.conv2 = Conv(32, 64, ksize=3, stride=2)
        self.C3_1 = C3(64, 64, n=1)
        self.conv3 = Conv(64, 128, ksize=3, stride=2) 

        self.C3_2 = C3(128, 128, n=2)
        self.conv4 = Conv(128, 256, ksize=3, stride=2) 
        
        self.C3_3 = C3(256, 256, n=3)
        self.conv5 = Conv(256, 512, ksize=3, stride=2)
        self.C3_4 =  C3(512, 512, n=1)

        self.sppf = SPPF(512, 512, kernel_size=5)

        # head
        self.conv6 = Conv(512, 256, ksize=1, stride=1)
        self.upsample = Upsample()
        self.concat = Concat(dimension=1)
        self.C3_5 = C3(512, 256, n=1, shortcut=False)

        self.conv7 = Conv(256, 128, ksize=1, stride=1) 

        self.C3_6 = C3(256, 128, n=3, shortcut=False)
        self.conv8 = Conv(128, 128, ksize=3, stride=2)

        self.C3_7 = C3(256, 256, n=1, shortcut=False) 
        self.conv9 = Conv(256, 256, ksize=3, stride=2)

        self.C3_8 = C3(512, 512, n=1, shortcut=False) 


    def forward(self , x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.C3_1(x)
      p3 = x = self.conv3(x)
      x = self.C3_2(x)
      p4 = x = self.conv4(x)
      x = self.C3_3(x)
      x = self.conv5(x)
      x = self.C3_4(x)
      x = self.sppf(x)
      p5 = x = self.conv6(x)
      x = self.upsample(x),
      x = self.concat([x,p4])
      x = self.C3_5(x)
      p6 = x = self.conv7(x)
      x = self.upsample(x)
      x = self.concat([x,p3])
      p7 = x = self.C3_6(x)
      x = self.conv8(x)
      x = self.concat([x,p6])
      p8 = x = self.C3_7(x)
      x = self.conv9(x)
      x = self.concat([x,p5])
      p9 = self.C3_8(x)

      #output
      p7 = M.Conv2d(128, (self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p7) # padding=0
      p7 = Reshape([self.image_shape[0]//8, self.image_shape[1]//8, self.anchors_per_location, self.num_class + 5])(p7)
      p8 = M.Conv2d(256, (self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p8)
      p8 = Reshape([self.image_shape[0]//16, self.image_shape[1]//16, self.anchors_per_location, self.num_class + 5])(p8)
      p9 = M.Conv2d(512, (self.num_class + 5) * self.anchors_per_location, kernel_size=1)(p9)
      p9 = Reshape([self.image_shape[0]//32, self.image_shape[1]//32, self.anchors_per_location, self.num_class + 5])(p9)
      
      return (p7, p8, p9)


# def gen_data():
#     while True:
#         image = tf.random.normal([2, 512, 512, 3])
#         p7 = tf.random.normal([2, 512 // 8, 512 // 8, 256])
#         p8 = tf.random.normal([2, 512 // 16, 512 // 16, 512])
#         p9 = tf.random.normal([2, 512 // 32, 512 // 32, 1024])
#         yield image, [p7, p8, p9]


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    yolo5s = Yolov5s(image_shape=(640, 640, 3),
                     batch_size=2,
                     num_class=30,
                     anchors_per_location=3)
    print(yolo5s.named_modules)
