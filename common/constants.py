import os
from os.path import join

curr_dir = os.path.dirname(os.path.abspath(__file__))
SMPL_MEAN_PARAMS = join(curr_dir, '../data/smpl_mean_params.npz')
SMPL_MODEL_DIR = join(curr_dir, '../data')

CROP_IMG_HEIGHT = 256
CROP_IMG_WIDTH = 192
CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
