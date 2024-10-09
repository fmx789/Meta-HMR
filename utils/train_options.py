import os
import json
import argparse
import numpy as np
from collections import namedtuple


class TrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', default='train_coco_loss', help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf,
                         help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=4, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        io.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_checkpoint', default=None,
                        help='Load a pretrained checkpoint at the beginning training')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--backbone', default='resnet', help='backbone: resnet/hrnet')
        train.add_argument('--model', default='cliff', help='select model: cliff/hmr')
        train.add_argument('--num_epochs', type=int, default=1600, help='Total number of training epochs')  ############
        train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")  ############
        train.add_argument('--batch_size', type=int, default=256, help='Batch size')  ############
        train.add_argument("--use_openpose2d", default=False, action='store_true',
                           help='Use OpenPose detect 2d for during evaluation')
        train.add_argument("--adapt_val_step", default=False, action='store_true',
                           help='adaptive val step, which will disable original val step')
        # train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network')
        train.add_argument('--syn_occ', default=False, action='store_true', help='Use Synthetic Occlusion')
        train.add_argument('--viz_debug', default=False, action='store_true', help='Use visualization for debugging')
        train.add_argument('--train_dataset', default=None,
                           help='which dataset to be trained on, if none, use full dataset')
        train.add_argument('--eval_dataset', default='3dpw', help='which dataset to be evaled on')
        train.add_argument('--debug_dataset', default='3dpw', help='which dataset to be debugged on')
        train.add_argument('--bbox_type', default='square',
                           help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')
        train.add_argument('--ALB', default=False, action='store_true', help='Use augmentation from bedlam')
        train.add_argument('--multilevel_crop_rand_prob', default=0.3, type=int,
                           help='probability to apply random body cropping')

        train.add_argument('--no_inner', action='store_true', help='not use inner step')

        train.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam optimizer')
        train.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
        train.add_argument('--outer_lr', type=float, default=1e-4, help='the outer lr')
        train.add_argument('--decay_lr', type=int, default=200000, help='decay outer learning rate every')
        train.add_argument('--eps_adam', type=float, default=1e-8, help='eps of adam optimzer')
        train.add_argument('--inner_lr', type=float, default=1e-5, help='the inner lr')
        train.add_argument('--aux_lr', type=float, default=1e-6, help='the lr of aux net')
        train.add_argument('--no_learn_loss', action='store_true', help='DO NOT learn loss function')
        train.add_argument('--first_order', action='store_true', help='use the first order gradient')
        train.add_argument('--inner_step', type=int, default=1, help='# of inner step')
        train.add_argument('--val_step', type=int, default=1, help='# of inner step')
        train.add_argument('--after_innerlr', type=float, default=2e-7, help='the inner lr')
        train.add_argument('--test_val_step', type=int, default=0, help='# of inner step')
        train.add_argument('--no_use_adam', action='store_true', help='not use_adam after 1 inner step')
        train.add_argument('--no_inner_step', action='store_true', help='use_adam after 1 inner step')

        train.add_argument('--use_gcn', default=False, action='store_true',
                           help='Use GCN for the second-stage feature integration')
        train.add_argument('--multi', default=False, action='store_true', help='Use Multi Stage feature regroup')
        train.add_argument('--use_latent', default=False, action='store_true', help='Use latent encoder')
        train.add_argument('--use_con', default=False, action='store_true',
                           help='Use contrastive loss to scatter latent features')
        train.add_argument('--use_parallel', default=False, action='store_true',
                           help='Use parallel stages for regression')
        train.add_argument('--use_extraviews', default=False, action='store_true',
                           help='Use parallel stages for regression')
        train.add_argument('--large_bbx', default=False, action='store_true', help='Use parallel stages for regression')
        train.add_argument('--use_matrix', default=False, action='store_true', help='Use matrix for regression')
        train.add_argument('--att_num', type=int, default=25, help='Number of attention feature')
        train.add_argument('--category_num', type=int, default=8,
                           help='Number of different training dataset categories')
        train.add_argument('--use_pseudo', default=False, action='store_true', help='Use pseudo images')
        train.add_argument('--with_raw', default=False, action='store_true', help='Use pseudo images')
        train.add_argument('--seperate', default=False, action='store_true', help='Use pseudo images')
        train.add_argument('--input_channels', type=int, default=3, help='Number of input images channels')
        train.add_argument('--rot_factor', type=float, default=30.,
                           help='Random rotation in the range [-rot_factor, rot_factor]')  ############
        train.add_argument('--noise_factor', type=float, default=0.4,
                           help='Randomly multiply pixel values with factor in the range [1-noise_factor, 1+noise_factor]')
        train.add_argument('--scale_factor', type=float, default=0.25,
                           help='Rescale bounding boxes by a factor of [1-scale_factor,1+scale_factor]')
        train.add_argument('--ignore_3d', default=False, action='store_true',
                           help='Ignore GT 3D data (for unpaired experiments')
        train.add_argument('--shape_loss_weight', default=0.5, type=float, help='Weight of per-vertex loss')
        train.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss')
        train.add_argument('--pose_loss_weight', default=1., type=float, help='Weight of SMPL pose loss')
        train.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss')
        train.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training')
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training')
        train.add_argument('--distill_loss_weight', type=float, default=1.,
                           help='Weight for distillation during training')
        train.add_argument('--run_smplify', default=False, action='store_true', help='Run SMPLify during training')
        train.add_argument('--smplify_threshold', type=float, default=100.,
                           help='Threshold for ignoring SMPLify fits during training')
        train.add_argument('--num_smplify_iters', default=100, type=int, help='Number of SMPLify iterations')

        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')  ############
        train.add_argument('--test_steps', type=int, default=100,
                           help='Testing frequency during training')  ############
        train.add_argument('--checkpoint_steps', type=int, default=100, help='Checkpoint saving frequency')
        train.add_argument('--test_epochs', type=int, default=50, help='Testing frequency(by epoch)')
        train.add_argument('--save_epochs', type=int, default=50, help='Saving frequency(by epoch)')
        train.add_argument('--test_start_epoch', type=int, default=0, help='Start testing from which epoch')

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)
        return

    def parse_args(self):
        """Parse input arguments."""
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, "r") as f:
                json_args = json.load(f)
                json_args = namedtuple("json_args", json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file.
        The default location is logs/expname/config.json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
