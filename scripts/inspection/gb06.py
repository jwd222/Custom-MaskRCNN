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

# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from mrcnn.model import DataGenerator, log
from mrcnn.visualize import display_images


# Custom Dataset Class
class GB06Dataset(utils.Dataset):

    def __init__(self, dataset_dir):
        self.data_dir = dataset_dir  # Store the directory path
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def load_dataset(self, data_dir, start_index=None, end_index=None):
        self.add_class("GB06Dataset", 1, "building")
        images_dir = os.path.join(data_dir, "Images")
        masks_dir = os.path.join(data_dir, "Json")
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
                "GB06Dataset",
                image_id=image_id,
                path=image_path,
                annotation=mask_path,
            )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID."""
        image_dir = os.path.join(self.data_dir, "Images", f"{image_id}.png")
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

        json_dir = os.path.join(self.data_dir, "Json")
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


class GB06Config(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """

    # Give the configuration a recognizable name
    NAME = "GB06"
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

    # MAX_GT_INSTANCES = 100
    # DETECTION_MAX_INSTANCES = 100

    # custom
    LEARNING_RATE = 0.001
    RUN_EAGERLY = None
    DETECTION_MIN_CONFIDENCE = 0.5
    WEIGHT_DECAY = 0.001
