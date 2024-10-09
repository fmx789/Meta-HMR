"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = '../datasets/h36m'
COCO_ROOT = '../datasets/coco'
PW3D_ROOT = '../datasets/pw3d'
MPII_ROOT = '../datasets/mpii'
MPI_INF_3DHP_ROOT = '../datasets/mpii3d'


# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/datasets'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_mosh_val_p1.npz'),
                    'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_mosh_val_p2_fixname_openpose.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_smpl_val_openpose.npz'),
                    'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid_openpose.npz'),
                    '3dpw': join(DATASET_NPZ_PATH, '3dpw_test_w2d_smpl3d_gender_openpose.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_mosh_train_fixname.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_smpl_w3d_train.npz'),
                    'coco': join(DATASET_NPZ_PATH, 'coco_2014_smpl_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train_name_revised.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_train_w2d_smpl3d_gender.npz'),
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                }

JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
