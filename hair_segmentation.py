"""
    Copyright (c) 2017 Matterport, Inc.
    Licensed under the MIT License (see LICENSE for details)
    Written by Waleed Abdulla

    # Train a new model starting from pre-trained COCO weights
    python3 run.py train --dataset=/path/to/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 run.py train --dataset=/path/to/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 run.py train --dataset=/path/to/dataset --weights=imagenet
    # Apply color mask to an image
    python3 run.py mask --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color mask to video using the last weights you trained
    python3 run.py mask --weights=last --video=<URL or path to file>
"""
"""
    Modified by JaeLin Joo
    2020.08.03
    # Add code that make the crop image backgroud transparent except the mask
    # Change image storage path
        * crop image : directory name - crop
        * crop_transparent image : directory name - crop_transparent

    Modified by Jaebeen Lee
    2020.08.06
    # Change image storage path
    # remove unnecessary code
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from PIL import Image
from mrcnn.config import Config
from mrcnn import model as modellib, utils


############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the Hair dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Hair"

    # Running on CPU
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Hair

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        # Add classes. We have only one class to add.
        self.add_class("Hair", 1, "Hair")
        dataset_dir = os.path.join(dataset_dir, subset)
        for filename in os.listdir(os.path.join(dataset_dir, 'photos')):
            if not filename.endswith('jpg'):  # Only jpg photos from dataset
                continue
            input_path = os.path.join(dataset_dir, 'photos', filename)
            img = cv2.imread(input_path)
            height, width = img.shape[:2]

            self.add_image(
                "Hair",  # for a single class just add the name here
                image_id=filename,  # use file name as a unique image id
                path=input_path,
                width=width, height=height)

    def load_mask(self, image_id):
        """Generate instance masks for an image from database.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "Hair":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        if info['id'].startswith('v'):  # If validation mask or training
            dataset_dir = os.path.join(args.dataset, 'val')
        else:
            dataset_dir = os.path.join(args.dataset, 'train')
        for maskf in os.listdir(os.path.join(dataset_dir, 'masks')):
            mname, png = os.path.splitext(maskf)
            iname, jpg = os.path.splitext(info['id'])
            if mname == iname:
                mask = cv2.imread(os.path.join(dataset_dir, 'masks', maskf))  # Reading mask data from dataset

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Hair":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def apply_mask(image, mask):
    os.environ["KERAS_BACKEND"] = "tensorflow"
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    blank = np.zeros(image.shape, dtype=np.uint8)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        crop = np.where(mask, image, blank).astype(np.uint8)
    else:
        crop = blank.astype(np.uint8)
    return crop


def detect_and_mask(model, image_path, number):
    assert image_path
    if image_path:
        import cv2
        # Run model detection and generate the mask
        print("Running on {}".format(image_path))
        # Read image
        image = cv2.imread(image_path)
        # Detect objects
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r = model.detect([image], verbose=1)[0]
        # Mask
        crop = apply_mask(image, r['masks'])
        # Save output
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        file_name = "crop.png"
        crop_path = './hair_crop'
        cv2.imwrite(os.path.join(crop_path, file_name), crop)

        # Make the background transparent except the mask
        MAKE_TRANSPARENT = True
        if (MAKE_TRANSPARENT):
            # open file
            img = Image.open(os.path.join(crop_path, file_name))
            # change image form RGBA
            img = img.convert("RGBA")
            datas = img.getdata()
            newData = []

            for item in datas:
                # If the pixel color is not black, add the corresponding area
                if (item[0] != 0 and item[1] != 0 and item[2] != 0):
                    newData.append(item)
                # If not make the background transparent
                else:
                    newData.append((255, 255, 255, 0))
            # input data
            img.putdata(newData)
            # save the image
            crop_transparent_path = './hair_crop_transparent'
            file_name2 = "crop_hair"+str(number)+".png"
            img.save(os.path.join(crop_transparent_path, file_name2))
    print("Saved to ", file_name)
    print("Saved to ", file_name2)


############################################################
#  Training
############################################################

def hair_segment(filename):
    Accuracy_Flag = True
    os.environ["KERAS_BACKEND"] = "tensorflow"

    # Root directory of the project
    ROOT_DIR = os.path.abspath("./")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    task_command = "mask"

    weight_path = "./model/"
    # weights_files = [weight_path + "mask_rcnn_hair_0027.h5", weight_path + "mask_rcnn_hair_0060.h5",
    #                      weight_path + "mask_rcnn_hair_0061.h5", weight_path + "mask_rcnn_hair_0140.h5",
    #                      weight_path + "mask_rcnn_hair_0145.h5", weight_path + "mask_rcnn_hair_0200.h5"]
    weights_files = [weight_path + "mask_rcnn_hair_0145.h5"]

    if(Accuracy_Flag):
        weights_files = [weight_path + "mask_rcnn_hair_0027.h5", weight_path + "mask_rcnn_hair_0060.h5",
                                               weight_path + "mask_rcnn_hair_0061.h5", weight_path + "mask_rcnn_hair_0140.h5",
                                               weight_path + "mask_rcnn_hair_0145.h5", weight_path + "mask_rcnn_hair_0200.h5"]
    image_file = filename

    for i in range(len(weights_files)):
        count = i
        # Validate arguments
        model_directory = weights_files[i]
        if task_command == "mask":
            print("Weights: ", model_directory)

        class InferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_directory)

        # Load weights
        print("Loading weights ", model_directory)
        model.load_weights(model_directory, by_name=True)
        detect_and_mask(model, image_file, count)
        #hair_crop_transparent/crop_hair.png