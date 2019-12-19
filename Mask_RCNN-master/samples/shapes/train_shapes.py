import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import json
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import imgaug as ia
import imgaug.augmenters as iaa

from itertools import groupby
from pycocotools import mask as maskutil

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # background + 20 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    STEPS_PER_EPOCH = 800

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()

coco = COCO("../../../pascal_train.json") # load training annotations

img_list = list(coco.imgs.keys())

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, st, ed, img_floder):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "aeroplane")
        self.add_class("shapes", 2, "bicycle")
        self.add_class("shapes", 3, "bird")
        self.add_class("shapes", 4, "boat")
        self.add_class("shapes", 5, "bottle")
        self.add_class("shapes", 6, "bus")
        self.add_class("shapes", 7, "car")
        self.add_class("shapes", 8, "cat")
        self.add_class("shapes", 9, "chair")
        self.add_class("shapes", 10, "cow")
        self.add_class("shapes", 11, "diningtable")
        self.add_class("shapes", 12, "dog")
        self.add_class("shapes", 13, "horse")
        self.add_class("shapes", 14, "motorbike")
        self.add_class("shapes", 15, "person")
        self.add_class("shapes", 16, "pottedplant")
        self.add_class("shapes", 17, "sheep")
        self.add_class("shapes", 18, "sofa")
        self.add_class("shapes", 19, "train")
        self.add_class("shapes", 20, "tvmonitor")

        # Add images
        for i in range(st, ed):
            imgIds = img_list[i] # Use the key above to retrieve information of the image
            img_info = coco.loadImgs(ids=imgIds)
            self.add_image("shapes", image_id=imgIds, path="{}{}".format(img_floder, img_info[0]['file_name']),
                           width=img_info[0]['width'], height=img_info[0]['height'])

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        imgIds = info['id'] # Use the key above to retrieve information of the image
        annids = coco.getAnnIds(imgIds=imgIds)
        anns = coco.loadAnns(annids)
        count = len(annids)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i in range(count):
          submask = coco.annToMask(anns[i])
          submask = submask.reshape(info['height'], info['width'], 1)
          mask[:, :, i:i+1] = submask
        
        # Map class names to class IDs.
        class_ids = np.array([anns[s]['category_id'] for s in range(len(annids))])
        return mask.astype(np.bool), class_ids.astype(np.int32)

img_folder = "../../../train_images/"
# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(135, 1214, img_folder)
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(0, 135, img_folder)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')

# Data augmentation
seq = iaa.Sometimes(0.833, iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.2), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45), # rotate by -45 to +45 degrees
        shear=(-16, 16), # shear by -16 to +16 degrees
        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )
], random_order=True)) # apply augmenters in random order

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=90, layers="all", augmentation=seq)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = "../../logs/shapes20191218T1531/mask_rcnn_shapes_0250.h5"
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

cocoGt = COCO("../../../test.json")

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

coco_dt = []
file_names = next(os.walk(IMAGE_DIR))[2]
for imgid in cocoGt.imgs:
  image = cv2.imread(IMAGE_DIR + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]

  # Run detection
  results = model.detect([image], verbose=1)
  r = results[0]
  n_instances = len(r['scores'])
  if n_instances > 0:
    for i in range(n_instances):
      # save information of the instance in a dictionary then append on coco_dt list
          pred = {}
          pred['image_id'] = imgid # this imgid must be same as the key of test.json
          pred['category_id'] = int(r['class_ids'][i])
          pred['segmentation'] = binary_mask_to_rle(r['masks'][:,:,i]) # save binary mask to RLE, e.g. 512x512 -> rle
          pred['score'] = float(r['scores'][i])
          coco_dt.append(pred)

with open("0856065_17.json", "w") as f:
    json.dump(coco_dt, f)