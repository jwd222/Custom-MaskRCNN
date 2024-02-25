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
from sklearn.metrics import PrecisionRecallDisplay
from tqdm import tqdm

######################################################
### GPU ####
#########################################
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# Root directory of the project
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from mrcnn.model import log

# from mrcnn.model import DataGenerator, log
from mrcnn.visualize import display_images

# Dataset Directory
DATASET_DIR = r"D:\projects\Building_sensing\building_construction_year\mask_rcnn_tf_2.3\Mask_RCNN\data\2009_train_data"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file

WEIGHT_DIR = os.path.join(ROOT_DIR, "weights")

ORG_MODEL_PATH2 = r"D:\projects\Building_sensing\building_construction_year\Mask_RCNN_TF_2.10\weights\old\maskrcnn_tuned_heads_300_per_150_noaug_34k.h5"
ORG_MODEL_PATH = r"D:\projects\Building_sensing\building_construction_year\mask_rcnn_tf_2.3\Mask_RCNN\weights\1.40\weights_7.0\tuned_on_34k_1.h5"

CLASS_NAMES = ["BG", "building"]


# Custom Dataset Class
class GB06Dataset(utils.Dataset):
    def load_dataset(self, dataset_dir, start_index=None, end_index=None):
        self.add_class("GB06Dataset", 1, "building")
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
                "GB06Dataset",
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
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


def fill_between(polygon):
    """
    Returns: a bool array
    """
    img = Image.new("1", (512, 512), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask


# Train dataset
dataset_train = GB06Dataset()
dataset_train.load_dataset(
    DATASET_DIR, start_index=0, end_index=3999
)  # Adjust the indices accordingly
dataset_train.prepare()

# Validation dataset
dataset_val = GB06Dataset()
dataset_val.load_dataset(
    DATASET_DIR, start_index=6000, end_index=6999
)  # Adjust the indices accordingly
dataset_val.prepare()


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
    TRAIN_ROIS_PER_IMAGE = 100

    USE_MINI_MASK = True

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20

    MAX_GT_INSTANCES = 250
    DETECTION_MAX_INSTANCES = 350

    # custom
    LEARNING_RATE = 0.001
    RUN_EAGERLY = None
    DETECTION_MIN_CONFIDENCE = 0.5
    WEIGHT_DECAY = 0.001


class InferenceConfig(GB06Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = GB06Config()


DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir=MODEL_DIR
    )
    model2 = modellib.MaskRCNN(
        mode="inference", config=inference_config, model_dir=MODEL_DIR
    )

# Load trained weights
print("Loading weights from ", ORG_MODEL_PATH)
model.load_weights(ORG_MODEL_PATH, by_name=True)
model2.load_weights(ORG_MODEL_PATH2, by_name=True)

# # Test on a random image
# image_id = random.choice(dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
#     dataset_val, inference_config, image_id, use_mini_mask=False
# )

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(
#     original_image,
#     gt_bbox,
#     gt_mask,
#     gt_class_id,
#     dataset_train.class_names,
#     figsize=(8, 8),
# )
# results = model.detect([original_image], verbose=1)

# r = results[0]
# visualize.display_instances(
#     original_image,
#     r["rois"],
#     r["masks"],
#     r["class_ids"],
#     dataset_val.class_names,
#     r["scores"],
#     # ax=get_ax(),
# )

# Compute VOC-Style mAP @ IoU=0.5
# Running on 100 images. Increase for better accuracy.


# image_ids = np.random.choice(dataset_val.image_ids, 500)
APs = []
y_true = []
probas_pred = []
y_true2 = []
probas_pred2 = []

for image_id in tqdm(dataset_val.image_ids[:10]):
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset_val, inference_config, image_id, use_mini_mask=False
    )
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    results2 = model2.detect([image], verbose=0)
    r = results[0]
    r2 = results2[0]

    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_bbox,
        gt_class_id,
        gt_mask,
        r["rois"],
        r["class_ids"],
        r["scores"],
        r["masks"],
        0.5,
    )

    gt_match2, pred_match2, overlaps2 = utils.compute_matches(
        gt_bbox,
        gt_class_id,
        gt_mask,
        r2["rois"],
        r2["class_ids"],
        r2["scores"],
        r2["masks"],
        0.5,
    )

    ##########  1   ##########
    # y_true.append( 0 if d < 0 else 1 for d in pred_match)
    y_true = y_true + list(0 if d < 0 else 1 for d in pred_match)
    # probas_pred.append(list(r["scores"]))
    probas_pred = probas_pred + list(r["scores"])

    ##########  2   ##########
    # y_true.append( 0 if d < 0 else 1 for d in pred_match)
    y_true2 = y_true2 + list(0 if d < 0 else 1 for d in pred_match2)
    # probas_pred.append(list(r["scores"]))
    probas_pred2 = probas_pred2 + list(r2["scores"])

    # # Compute AP
    # AP, precisions, recalls, overlaps = utils.compute_ap(
    #     gt_bbox,
    #     gt_class_id,
    #     gt_mask,
    #     r["rois"],
    #     r["class_ids"],
    #     r["scores"],
    #     r["masks"],
    # )
    # APs.append(AP)


display = PrecisionRecallDisplay.from_predictions(y_true, probas_pred, name="LinearSVC")
_ = display.ax_.set_title("Single Class Precision-Recall curve")

display2 = PrecisionRecallDisplay.from_predictions(
    y_true2, probas_pred2, name="LinearSVC"
)
_ = display2.ax_.set_title("Single Class Precision-Recall curve")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots

# Plot the first Precision-Recall curve on the first subplot
display.plot(ax=axes[0], name="Model 1")

# Plot the second Precision-Recall curve on the second subplot
display2.plot(ax=axes[1], name="Model 2")

# Set titles for each subplot
axes[0].set_title("Precision-Recall Curve")
axes[1].set_title("Precision-Recall Curve")

# Set a single legend for both subplots
axes[0].legend(loc="lower left")
axes[1].legend(loc="lower left")

plt.show()


print(" ")

# print("mAP: ", np.mean(APs))
