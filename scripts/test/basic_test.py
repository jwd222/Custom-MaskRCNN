# Mask R-CNN Test

import math
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils, visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images", "2009_test")
## Configurations

# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.
from mrcnn.config import Config

CLASS_NAMES = ["BG", "building"]


class SpaceNetConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "SpaceNet"
    BACKBONE = "resnet50"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 building

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.25, 1, 4]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 100

    USE_MINI_MASK = True

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 30

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10

    # MAX_GT_INSTANCES = 300
    # DETECTION_MAX_INSTANCES = 200

    # custom
    LEARNING_RATE = 0.01
    RUN_EAGERLY = None
    DETECTION_MIN_CONFIDENCE = 0.6
    WEIGHT_DECAY = 0.01


config = SpaceNetConfig()
config.display()
## Create Model and Load Trained Weights
# Create model object in inference mode.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Get path to saved weights
# Either set a specific path or find last trained weights
# pretrained_model_path = os.path.join(ROOT_DIR, "mask_rcnn_spacenet_0151.h5")

## original
# model_path = os.path.join(
#     ROOT_DIR, r"weights\maskrcnn_tuned_heads_300_per_150_noaug_34k.h5"
# )

# for testing a random model
model_path = r"D:\projects\Building_sensing\building_construction_year\mask_rcnn_tf_2.3\Mask_RCNN\weights\1.40\weights_7.0\tuned_on_34k_5.h5"

# model_path = model.find_last()
# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

## Run Object Detection
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

output_path = r"D:\projects\Building_sensing\building_construction_year\Mask_RCNN_TF_2.10\images\detected_images\33"
# if folder doesnt exits, make it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Run detection for all images
for image_name in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(
        image,
        r["rois"],
        r["masks"],
        r["class_ids"],
        CLASS_NAMES,
        r["scores"],
        save_fig_path=os.path.join(output_path, image_name),
    )
    print(" ")
print(" ")
