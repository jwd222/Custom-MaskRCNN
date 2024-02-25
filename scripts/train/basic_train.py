import json
import os
import random
import sys
import time

import cv2
import imgaug
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from mrcnn.model import DataGenerator, log
from mrcnn.visualize import display_images

# Root directory of the project

# DATASET_DIR = os.path.abspath("test_dataset/train_data") #Sadece bu kısmı değiştir yeter
DATASET_DIR = os.path.join(ROOT_DIR, "data", "main_datasets", "example_train_data")

# print(ROOT_DIR)

# print(DATASET_DIR)

# Import Mask RCNN
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file

WEIGHT_DIR = os.path.join(ROOT_DIR, "weights")
ORG_MODEL_PATH = os.path.join(WEIGHT_DIR, "others", "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# print(MODEL_DIR)

# ROOT_DIR
# print(ORG_MODEL_PATH)

CLASS_NAMES = ["BG", "building"]


class SpaceNetConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "SpaceNet"
    BACKBONE = "resnet101"
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
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.25, 1, 4]

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    USE_MINI_MASK = True

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 10

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    MAX_GT_INSTANCES = 100
    DETECTION_MAX_INSTANCES = 100

    # custom
    LEARNING_RATE = 0.0001
    RUN_EAGERLY = None
    DETECTION_MIN_CONFIDENCE = 0.5


config = SpaceNetConfig()
# config.display()


class SpaceNetDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, start_index=None, end_index=None):
        self.add_class("SpaceNetDataset", 1, "building")
        images_dir = os.path.join(dataset_dir, "Images")
        masks_dir = os.path.join(dataset_dir, "Json")
        # Get all file names in the Images directory
        image_names = sorted(os.listdir(images_dir))

        # Optionally limit the range of images to load
        if start_index is not None and end_index is not None:
            image_names = image_names[start_index : end_index + 1]

        for image_name in image_names:
            # Get the image ID without relying on the ".png" extension
            image_id, _ = os.path.splitext(image_name)
            # Construct paths
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(masks_dir, image_name)
            # Add to the dataset
            self.add_image(
                "SpaceNetDataset",
                image_id=image_id,
                path=image_path,
                annotation=mask_path,
            )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID."""
        image_dir = os.path.join(DATASET_DIR, "Images", f"{image_id}.png")
        im = Image.open(image_dir)
        # Check the number of channels
        if im.mode == "RGBA":
            # If the image has an alpha channel, remove it
            im = im.convert("RGB")
        return np.asarray(im)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID."""
        masks = np.zeros((512, 512))

        json_dir = os.path.join(DATASET_DIR, "Json")
        with open(os.path.join(json_dir, str(image_id) + ".json")) as f:
            data = json.load(f)
            allBuildings = data["features"]

            for building in allBuildings:
                buildings_polygon = building["coordinates"]  # if inside a polygon
                tuple_polygon = [tuple(point) for point in buildings_polygon]
                maske = fill_between(tuple_polygon)
                masks = np.dstack((masks, maske))

        if masks.shape != (512, 512):
            masks = masks[:, :, 1:]
            class_ids = np.asarray([1] * masks.shape[2])
        else:
            class_ids = np.ones((1))
            masks = masks.reshape((512, 512, 1))
        return masks.astype(np.bool_), class_ids.astype(np.int32)


# util function
def fill_between(polygon):
    """
    Returns: a bool array
    """
    img = Image.new("1", (512, 512), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask


# print("dataset directory is: ", DATASET_DIR)

# Train dataset
dataset_train = SpaceNetDataset()
dataset_train.load_dataset(
    DATASET_DIR, start_index=0, end_index=80
)  # Adjust the indices accordingly
dataset_train.prepare()

# Validation dataset
dataset_val = SpaceNetDataset()
dataset_val.load_dataset(
    DATASET_DIR, start_index=81, end_index=100
)  # Adjust the indices accordingly
dataset_val.prepare()

print(" ")

"""
inspect data
"""
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info["name"]))


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

"""
Creating model in training mode
"""
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

weights_path = ORG_MODEL_PATH
# model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(
    weights_path,
    by_name=True,
    # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
)

# Load the last model you trained and continue training
# model.load_weights(model.find_last(), by_name=True)


start_train = time.time()
print("Training head layers")
model.train(
    dataset_train,
    dataset_val,
    learning_rate=config.LEARNING_RATE,
    epochs=10,
    layers="heads",
    # augmentation = imgaug.augmenters.Sequential([
    # imgaug.augmenters.Fliplr(1),
    # imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
    # imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
    # imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
    # imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
    # ]
    # )
)
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f"Total training time: {minutes} minutes")

# save model
model_path = os.path.join(
    ROOT_DIR, "weights_5.0", "maskrcnn_tuned_4+_300_per_300_noaug_34k.h5"
)
model.keras_model.save_weights(model_path)
