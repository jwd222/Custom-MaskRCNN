import json
import os

# import pdb
# import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from mrcnn.model import log
import imgaug
import os
import sys

import numpy as np


import cv2
from PIL import Image
import skimage.io

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize



# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# DATASET_DIR = os.path.abspath("test_dataset/train_data") #Sadece bu kısmı değiştir yeter
DATASET_DIR = os.path.join(r"E:\Projects\GB06_building_sensing\data\Oldampt_new_train_data")

# print(ROOT_DIR)

# print(DATASET_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
ORG_MODEL_PATH = os.path.join(ROOT_DIR, r"weights_5.0\maskrcnn_tuned_heads_300_per_150_noaug_34k.h5")
# Download COCO trained weights from Releases if needed
# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models/mask_rcnn_coco.h5")
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# print(MODEL_DIR)

ROOT_DIR
# print(ORG_MODEL_PATH)

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
    TRAIN_ROIS_PER_IMAGE = 32

    USE_MINI_MASK = True
    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    LEARNING_RATE = 0.0001

    MAX_GT_INSTANCES=250
    DETECTION_MAX_INSTANCES=350
    
config = SpaceNetConfig()
# config.display()

# print(DATASET_DIR)

class SpaceNetDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, start_index=None, end_index=None):
        self.add_class("SpaceNetDataset", 1, "building")
        images_dir = os.path.join(dataset_dir, "Images")
        masks_dir = os.path.join(dataset_dir, "Json")
        # Get all file names in the Images directory
        image_names = sorted(os.listdir(images_dir))

        # Optionally limit the range of images to load
        if start_index is not None and end_index is not None:
            image_names = image_names[start_index:end_index+1]

        for image_name in image_names:
            # Get the image ID without relying on the ".png" extension
            image_id, _ = os.path.splitext(image_name)
            # Construct paths
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(masks_dir, image_name)
            # Add to the dataset
            self.add_image('SpaceNetDataset', image_id=image_id, path=image_path, annotation=mask_path)


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID."""
        image_dir = os.path.join(DATASET_DIR, "Images", "{}.png".format(image_id))
        im = Image.open(image_dir)
        # Check the number of channels
        if im.mode == 'RGBA':
            # If the image has an alpha channel, remove it
            im = im.convert('RGB')
        return np.asarray(im)

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
        masks = np.zeros((512,512))
  
        json_dir = os.path.join(DATASET_DIR, "Json")
        with open(os.path.join(json_dir, str(image_id) + ".json")) as f:
            data = json.load(f)
            allBuildings = data['features']
              
            for building in allBuildings:   
                buildings_polygon = building['coordinates'] # if inside a polygon 
                tuple_polygon = [tuple(point) for point in buildings_polygon]
                maske = fill_between(tuple_polygon)
                masks = np.dstack((masks,maske))

        if masks.shape != (512,512):
            masks = masks[:,:,1:]
            class_ids = np.asarray([1]*masks.shape[2])
        else:
            class_ids=np.ones((1))
            masks = masks.reshape((512,512,1))
        return masks.astype(np.bool_), class_ids.astype(np.int32)


# More funtions
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def fill_between(polygon):
    """
    Returns: a bool array
    """
    img = Image.new('1', (512, 512), False)
    ImageDraw.Draw(img).polygon(polygon, outline=True, fill=True)
    mask = np.array(img)
    return mask

def load_image(image):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def imageDetails(image_path):
    original_image = skimage.io.imread(image_path)

    # Print basic information
    print("Image Path:", image_path)
    print("Image Shape:", original_image.shape)
    print("Image Data Type:", original_image.dtype)
    print("Image Min Value:", np.min(original_image))
    print("Image Max Value:", np.max(original_image))

    # If the image is a color image (RGB), print more details
    if original_image.ndim == 3 and original_image.shape[2] == 3:
        print("Color Image")
        print("Red Channel Range:", np.min(original_image[:, :, 0]), "-", np.max(original_image[:, :, 0]))
        print("Green Channel Range:", np.min(original_image[:, :, 1]), "-", np.max(original_image[:, :, 1]))
        print("Blue Channel Range:", np.min(original_image[:, :, 2]), "-", np.max(original_image[:, :, 2]))

    # If the image has an alpha channel, print alpha channel details
    if original_image.ndim == 3 and original_image.shape[2] == 4:
        print("Alpha Channel Image")
        print("Alpha Channel Range:", np.min(original_image[:, :, 3]), "-", np.max(original_image[:, :, 3]))


# Functions for proper testing
def process_image(image_path, output_folder):
    # Load image from path
    original_image = skimage.io.imread(image_path)
    
    # Check if the image has an alpha channel (4 channels)
    if original_image.shape[-1] == 4:
        # Remove the alpha channel to convert it to RGB (3 channels)
        original_image = original_image[:, :, :3]

    # Printing image details
    # imageDetails(image_path)
    # print([original_image])
    
    # Get results
    results = model.detect([original_image], verbose=1)
    r = results[0]
    # print(r["masks"].shape, "---------------------------------------")
    
    # Extract the filename from the input image path
    filename = os.path.basename(image_path)
    
    # Create the "output_masks" folder in the output folder if it doesn't exist
    output_masks_folder = os.path.join(output_folder, "maskrcnn_output_masks")
    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)
    
    # Construct the output path for the validation image with "_mask" at the end
    validation_image_name = os.path.splitext(filename)[0] + "_mask.png"
    validation_image_path = os.path.join(output_masks_folder, validation_image_name)
    
    # Assuming you have the original image and mask array
    mask_array = r["masks"]

    # Ensure that the original image and mask array have the same dimensions
    assert original_image.shape[:2] == mask_array.shape[:2], "Mismatched dimensions between image and mask"

    # Initialize a black canvas with the same dimensions as the original image
    validation_image = Image.fromarray(np.zeros_like(original_image).astype(np.uint8))

    # Iterate through the masks and overlay them on the canvas
    for i in range(mask_array.shape[2]):  # Loop through the number of masks
        mask = (mask_array[:, :, i] * 255).astype(np.uint8)  # Get the individual mask, ensure it's uint8
        # Set the pixels within the mask to white
        validation_image = Image.fromarray(np.array(validation_image) | (np.stack([mask] * 3, axis=-1)))

    # Save the validation image with the same filename as the input image
    validation_image.save(validation_image_path)
    
def test(input_path, model_path):
    # Load trained weights
    print("Loading weights from ", model_path)
    # model.load_weights(model_path, by_name=True)
    model.load_weights(model_path, by_name=True)
    
    for image_name in os.listdir(input_path):
        if image_name.endswith(".png"):
            image_path = os.path.join(input_path, image_name)
            # print(image_path)
            
            output_path = os.path.join(input_path, "test_masks_3", "mask_rcnn_tuned_head_layers_on_300_epochs_with_augmentation") 
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                
            process_image(image_path, output_path)
                
            output_image = os.path.join(output_path, image_name) 
            # print(output_image)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            
            r = model.detect([image], verbose=0)
            r = r[0]
            mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'],
                                  save_fig_path=output_image)

# print("dataset directory is: ", DATASET_DIR)

# Train dataset
dataset_train = SpaceNetDataset()
dataset_train.load_dataset(DATASET_DIR, start_index=0, end_index=5496)  # Adjust the indices accordingly
dataset_train.prepare()

# Validation dataset
dataset_val = SpaceNetDataset()
dataset_val.load_dataset(DATASET_DIR, start_index=5496, end_index=6870)  # Adjust the indices accordingly
dataset_val.prepare()

'''
Load and disply sample
'''
# image_id = 6
# image = dataset_train.load_image(image_id)
# mask, class_ids = dataset_train.load_mask(image_id)
# visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

'''
Creating model in training mode
'''
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)


weights_path = ORG_MODEL_PATH
# model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.load_weights(weights_path, by_name=True)

# Load the last model you trained and continue training
# model.load_weights(model.find_last(), by_name=True)


start_train = time.time()
print("Training head layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=200,
            layers='4+',
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
print("Total training time: {} minutes".format(minutes))

# save model
model_path = os.path.join(ROOT_DIR, "weights_5.0", 'maskrcnn_tuned_4+_300_per_300_noaug_34k.h5')
model.keras_model.save_weights(model_path)


'''
Create model in inference mode
'''
# class InferenceConfig(SpaceNetConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# inference_config = InferenceConfig()

# model = modellib.MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=MODEL_DIR)
# '''
# Proper testing
# '''
# input_path = os.path.join(ROOT_DIR, r"test_dataset\final_test")
# model_path = os.path.join(ROOT_DIR, r"weights_2.0\mask_rcnn_tuned_head_layers_on_300_epochs_with_augmentation.h5")

# test(input_path, model_path)

# '''
# Test on random sample
# '''
# # Test on a random image
# image_id = 2
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, inference_config, 
#                            image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
#                             dataset_train.class_names, figsize=(8, 8))

# results = model.detect([original_image], verbose=1)
# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                             dataset_val.class_names, r['scores'], ax=get_ax())



'''
Confusion Matrix
'''
# config = inference_config