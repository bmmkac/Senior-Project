from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

class BboxConvertor:

    def __init__(self,width,height,bboxs):
        self.bbox = []
        self.h = height
        self.w = width
        
        for item in bboxs:
            if item[2] == 0:
                #load words boxes
                new_box = item[0]
                # set new_box x,y coord = 0,0
                self.bbox.append(new_box)
            else:
                #load image boxes
                new_box = item[0]
                # set new_box x,y coord = width/2,height/2
                self.bbox.append(new_box)

    def make_im(self):
        img = np.zeros([100,100,3],dtype=np.uint8)
        img.fill(255)
        image = utils.draw_bbox(img, self.bboxs)
        return image
