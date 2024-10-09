from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import os
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt

from datasets import BaseDataset, MixedDataset, MixedDataset_wo3dpw
from eval_cliff import run_evaluation
from models import SMPL
from models.meta_models import build_model
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, cam_crop2full
from utils.imutils import flip_img
from utils.renderer import Renderer
from utils import BaseTrainer

import config
import constants
from copy import deepcopy
from models.maml import gradient_update_parameters


class Trainer_cliff(BaseTrainer):

    def init_fn(self):
        # train with single dataset
        # if self.options.train_dataset is not None:
        #     self.train_ds = BaseDataset(self.options, self.options.train_dataset, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
       
       # train with multiple datasets
        self.train_ds = MixedDataset_wo3dpw(self.options, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)
        # self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, use_augmentation=True, is_train=True, bbox_type=self.options.bbox_type)

        # self.eval_dataset = BaseDataset(None, self.options.eval_dataset, is_train=False,
        #                                 bbox_type=self.options.bbox_type)
        self.eval_dataset_3dpw = BaseDataset(None, '3dpw', is_train=False,
                                             bbox_type=self.options.bbox_type)
        self.eval_dataset_h36m = BaseDataset(None, 'h36m-p2', is_train=False,
                                             bbox_type=self.options.bbox_type)
        
        self.outer_lr = self.options.outer_lr
        self.inner_lr = self.options.inner_lr
        self.inner_steps = self.options.inner_step
        self.val_steps = self.options.val_step
        self.first_order = self.options.first_order
        if self.options.no_inner:
            self.val_steps = 1
            self.inner_steps = 0
        print('######################### inner lr is %.8f #############################' % (self.inner_lr))
        print('######################### outer lr is %.8f, decay_lr %d #############################' % (
            self.outer_lr, self.options.decay_lr))
        print('######################### use first order %s #############################' % (self.first_order))
        print('######################### use backbone %s #############################' % (self.options.backbone))
        self.model = build_model(config.SMPL_MEAN_PARAMS, pretrained=True, backbone=self.options.backbone, name='cliff_meta',
                                 bbox_type=self.options.bbox_type).to(self.device)
        self.model_aux = build_model(config.SMPL_MEAN_PARAMS, pretrained=True, backbone=self.options.backbone, name='cliff_aux',
                                        bbox_type=self.options.bbox_type).to(self.device)
        self.params = OrderedDict(self.model.meta_named_parameters())

        if not self.options.no_learn_loss:
            print('------------------ with aux net, learn rate %.8f-----------------------' % self.options.aux_lr)
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters(), 'lr': self.outer_lr,
                 'betas': [self.options.beta1, self.options.beta2],
                 'eps': self.options.eps_adam},
                {'params': self.model_aux.parameters(), 'lr': self.options.aux_lr}
            ])

        else:
            print('------------------ without aux net -----------------------')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr,
                                              betas=[self.options.beta1, self.options.beta2], eps=self.options.eps_adam)

        
        self.joints_idx = 25
        self.joints_num = 49
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                            batch_size=self.options.batch_size,
                            create_transl=False).to(self.device)

        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model, 'model_aux': self.model_aux}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.render_focal_length = constants.FOCAL_LENGTH
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.use_pseudo = self.options.use_pseudo
        self.bbox_type = self.options.bbox_type
        if self.bbox_type == 'square':
            self.crop_w = constants.IMG_RES
            self.crop_h = constants.IMG_RES
            self.bbox_size = constants.IMG_RES
        elif self.bbox_type == 'rect':
            self.crop_w = constants.IMG_W
            self.crop_h = constants.IMG_H
            self.bbox_size = constants.IMG_H
        self.J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

    def compute_keypoints2d_loss_cliff(
            self,
            pred_keypoints3d: torch.Tensor,
            pred_cam: torch.Tensor,
            gt_keypoints2d: torch.Tensor,
            camera_center: torch.Tensor,
            focal_length: torch.Tensor,
            crop_trans,
            img_h,
            img_w,
            img,
            dataset,
            img_name,
            is_flipped,
            crop_center,
            gt_keypoints2d_full,
            rot,
            viz,
            dbg_dataset, ):
        """Compute loss for 2d keypoints."""

        gt_keypoints2d = gt_keypoints2d[:, :, :].float()

        device = gt_keypoints2d.device
        batch_size = pred_keypoints3d.shape[0]

        pred_keypoints2d_full = perspective_projection(
            pred_keypoints3d,
            rotation=torch.eye(3, device=device).unsqueeze(0).expand(
                batch_size, -1, -1),
            translation=pred_cam,
            focal_length=focal_length,
            camera_center=camera_center)

        pred_keypoints2d = torch.cat(
            (pred_keypoints2d_full, torch.ones(batch_size, self.joints_num, 1).to(device)),
            dim=2)
        # trans @ pred_keypoints2d2
        pred_keypoints2d_bbox = torch.einsum('bij,bkj->bki', crop_trans,
                                             pred_keypoints2d)

        pred_keypoints2d_bbox[:, :, 0] = 2. * pred_keypoints2d_bbox[:, :, 0] / self.crop_w - 1.
        gt_keypoints2d[:, :, 0] = 2. * gt_keypoints2d[:, :, 0] / self.crop_w - 1.
        pred_keypoints2d_bbox[:, :, 1] = 2. * pred_keypoints2d_bbox[:, :, 1] / self.crop_h - 1.
        gt_keypoints2d[:, :, 1] = 2. * gt_keypoints2d[:, :, 1] / self.crop_h - 1.

        loss = self.keypoint_loss(pred_keypoints2d_bbox.float(), gt_keypoints2d.float(),
                                  self.options.openpose_train_weight,
                                  self.options.gt_train_weight)

        return loss

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """

        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight

        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, dataset=None, viz=False,
                         dbg_dataset='3dpw', use_model=False, use_pseudo=False):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, self.joints_idx:, :]
        if use_model:
            conf = torch.ones_like(gt_keypoints_3d[:, :, -1].unsqueeze(-1), device=self.device)
            gt_keypoints_3d = gt_keypoints_3d.clone()
        elif use_pseudo:
            conf = torch.ones_like(gt_keypoints_3d[:, :, -1].unsqueeze(-1)[:, self.joints_idx:, :], device=self.device)
            gt_keypoints_3d = gt_keypoints_3d.clone()[:, self.joints_idx:, :]
        else:
            conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
            gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()


        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl, dataset=None, viz=False, dbg_dataset='3dpw'):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]

        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, dataset=None, use_pseudo=False):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        if use_pseudo:
            gt_rotmat_valid = gt_pose[has_smpl == 1]
        else:
            gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
        
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
        if len(pred_betas_valid) > 0:
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch, cur_epoch, cur_step):
        self.model.train()
        self.model_aux.train()

        # Get data from the batch
        images = input_batch['img']  # input image
        img_name = input_batch['imgname']
        gt_keypoints_2d_full = input_batch['keypoints_full'][:,:self.joints_num]
        gt_keypoints_2d = input_batch['keypoints'][:,:self.joints_num]  # 2D keypoints
        gt_pose = input_batch['pose']  # SMPL pose parameters
        gt_betas = input_batch['betas']  # SMPL beta parameters
        gt_joints = input_batch['pose_3d'][:,:self.joints_num]  # 3D pose
        has_smpl = input_batch['has_smpl'].byte()  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte()  # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped']  # flag that indicates whether image was flipped during data augmentation
        crop_trans = input_batch['crop_trans']
        full_trans = input_batch['full_trans']
        inv_trans = input_batch['inv_trans']
        rot_angle = input_batch['rot_angle']  # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name']  # name of the dataset the image comes from
        indices = input_batch['sample_index']  # index of example inside its dataset
        batch_size = images.shape[0]
        bbox_info = input_batch['bbox_info']
        center, scale, focal_length = input_batch['center'], input_batch['scale'], input_batch['focal_length'].float()

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, 0] = 0.5 * self.crop_w * (gt_keypoints_2d_orig[:, :, 0] + 1)
        gt_keypoints_2d_orig[:, :, 1] = 0.5 * self.crop_h * (gt_keypoints_2d_orig[:, :, 1] + 1)

        # Feed images in the network to predict camera and SMPL parameters
        # pred_rotmat, pred_betas, pred_camera = self.model(images, bbox_info)
        img_h, img_w = input_batch['img_h'].view(-1, 1), input_batch['img_w'].view(-1, 1)
        full_img_shape = torch.hstack((img_h, img_w))
        camera_center = torch.hstack((img_w, img_h)) / 2
        loss_shape_outer = torch.tensor(0., device=self.device)
        loss_keypoints_outer = torch.tensor(0., device=self.device)
        loss_keypoints_3d_outer = torch.tensor(0., device=self.device)
        loss_regr_pose_outer = torch.tensor(0., device=self.device)
        loss_regr_betas_outer = torch.tensor(0., device=self.device)
        loss_cam_outer = torch.tensor(0., device=self.device)

        group_size = batch_size
        group_num = int(batch_size / group_size)

        for i in range(group_num):
            task_id = range(i * group_size, (i + 1) * group_size)
            images_tr = images[task_id]
            bbox_info_tr = bbox_info[task_id]
            center_tr = center[task_id]
            camera_center_tr = camera_center[task_id]
            full_img_shape_tr = full_img_shape[task_id]
            scale_tr = scale[task_id]
            focal_length_tr = focal_length[task_id]
            params_cp = self.params

            # inner loop
            for in_step in range(self.inner_steps):
                pred_rotmat_inner, pred_betas_inner, pred_camera_inner = self.model(images_tr, bbox_info_tr,
                                                                                    params=params_cp)

                pred_rotmat_aux, pred_betas_aux, pred_camera_aux = self.model_aux(images_tr, bbox_info_tr)

                pred_output_inner = self.smpl(betas=pred_betas_inner, body_pose=pred_rotmat_inner[:, 1:],
                                              global_orient=pred_rotmat_inner[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices_inner = pred_output_inner.vertices
                pred_joints_inner = pred_output_inner.joints[:,:self.joints_num]

                pred_output_aux = self.smpl(betas=pred_betas_aux, body_pose=pred_rotmat_aux[:, 1:],
                                            global_orient=pred_rotmat_aux[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices_aux = pred_output_aux.vertices
                pred_joints_aux = pred_output_aux.joints[:,:self.joints_num]

                pred_cam_full_inner = cam_crop2full(pred_camera_inner, center_tr, scale_tr, full_img_shape_tr,
                                                    focal_length_tr).to(torch.float32)

                # Compute loss on SMPL parameters
                loss_regr_pose_inner, loss_regr_betas_inner = self.smpl_losses(pred_rotmat_inner, pred_betas_inner,
                                                                                pred_rotmat_aux, pred_betas_aux,
                                                                                has_smpl[task_id],
                                                                                dataset=dataset_name,
                                                                                use_pseudo=True)


                # Compute 2D reprojection loss for the keypoints
                loss_keypoints_inner = self.compute_keypoints2d_loss_cliff(
                    pred_joints_inner,
                    pred_cam_full_inner,
                    gt_keypoints_2d.clone()[task_id],
                    camera_center_tr,
                    focal_length_tr,
                    crop_trans.clone()[task_id],
                    img_h=img_h[task_id],
                    img_w=img_w[task_id],
                    img=images_tr,
                    dataset=dataset_name,
                    img_name=img_name[i * group_size:(i + 1) * group_size],
                    is_flipped=is_flipped[task_id],
                    crop_center=center_tr,
                    gt_keypoints2d_full=gt_keypoints_2d_full.clone()[task_id],
                    rot=rot_angle[task_id],
                    viz=self.options.viz_debug,
                    dbg_dataset=self.options.debug_dataset,
                )

                # Compute 3D keypoint loss
                loss_keypoints_3d_inner = self.keypoint_3d_loss(pred_joints_inner, pred_joints_aux,
                                                                has_pose_3d[task_id], dataset=dataset_name,
                                                                viz=self.options.viz_debug,
                                                                dbg_dataset=self.options.debug_dataset, use_pseudo=True)

                # Per-vertex loss for the shape
                loss_shape_inner = self.shape_loss(pred_vertices_inner, pred_vertices_aux,
                                                    has_smpl[task_id],
                                                    dataset=dataset_name,
                                                    viz=self.options.viz_debug,
                                                    dbg_dataset=self.options.debug_dataset)

                # Compute total loss
                # The last component is a loss that forces the network to predict positive depth values
                inner_loss = self.options.shape_loss_weight * loss_shape_inner + \
                             self.options.keypoint_loss_weight * loss_keypoints_inner + \
                             self.options.keypoint_loss_weight * loss_keypoints_3d_inner + \
                             self.options.pose_loss_weight * loss_regr_pose_inner + self.options.beta_loss_weight * loss_regr_betas_inner + \
                             ((torch.exp(-pred_camera_inner[:, 0] * 10)) ** 2).mean()

                self.model.zero_grad()
                params_cp = gradient_update_parameters(self.model, inner_loss, params=params_cp,
                                                       step_size=self.inner_lr, first_order=self.first_order)

            # outer loop
            pred_rotmat, pred_betas, pred_camera = self.model(images_tr, bbox_info_tr, params=params_cp)

            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                    global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints[:,:self.joints_num]

            pred_cam_crop = torch.stack([pred_camera[:, 1],
                                         pred_camera[:, 2],
                                         2 * self.render_focal_length / (self.bbox_size * pred_camera[:, 0] + 1e-9)],
                                        dim=-1)

            pred_cam_full = cam_crop2full(pred_camera, center_tr, scale_tr, full_img_shape_tr,
                                          focal_length_tr).to(torch.float32)

            loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, gt_pose[task_id],
                                                                gt_betas[task_id][:, :10],
                                                                has_smpl[task_id],
                                                                dataset=dataset_name, use_pseudo=False)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints = self.compute_keypoints2d_loss_cliff(
                pred_joints,
                pred_cam_full,
                gt_keypoints_2d.clone()[task_id],
                camera_center_tr,
                focal_length_tr,
                crop_trans.clone()[task_id],
                img_h=img_h[task_id],
                img_w=img_w[task_id],
                img=images_tr,
                dataset=dataset_name,
                img_name=img_name[i * group_size:(i + 1) * group_size],
                is_flipped=is_flipped[task_id],
                crop_center=center_tr,
                gt_keypoints2d_full=gt_keypoints_2d_full.clone()[task_id],
                rot=rot_angle[task_id],
                viz=self.options.viz_debug,
                dbg_dataset=self.options.debug_dataset,
            )

            loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints[task_id],
                                                      has_pose_3d[task_id], dataset=dataset_name,
                                                      viz=self.options.viz_debug,
                                                      dbg_dataset=self.options.debug_dataset, use_pseudo=False)

            # Per-vertex loss for the shape
            loss_shape = self.shape_loss(pred_vertices, gt_vertices[task_id],
                                            has_smpl[task_id], dataset=dataset_name,
                                            viz=self.options.viz_debug, dbg_dataset=self.options.debug_dataset)

            loss_cam = ((torch.exp(-pred_camera[:, 0] * 10)) ** 2).mean()

            loss_shape_outer += loss_shape
            loss_keypoints_outer += loss_keypoints
            loss_keypoints_3d_outer += loss_keypoints_3d
            loss_regr_pose_outer += loss_regr_pose
            loss_regr_betas_outer += loss_regr_betas
            loss_cam_outer += loss_cam

        loss_shape_outer.div_(group_num)
        loss_keypoints_outer.div_(group_num)
        loss_keypoints_3d_outer.div_(group_num)
        loss_regr_pose_outer.div_(group_num)
        loss_regr_betas_outer.div_(group_num)
        loss_cam_outer.div_(group_num)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape_outer + \
               self.options.keypoint_loss_weight * loss_keypoints_outer + \
               self.options.keypoint_loss_weight * loss_keypoints_3d_outer + \
               self.options.pose_loss_weight * loss_regr_pose_outer + self.options.beta_loss_weight * loss_regr_betas_outer + \
               loss_cam_outer
        loss *= 60
        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'pred_camera': pred_camera.detach(),
                  'pred_cam_crop': pred_cam_crop.detach(),
                  'pred_cam_full': pred_cam_full.detach(),
                  'dataset': dataset_name,
                  'img_name': img_name[-group_size:],
                  'gt_vertices': gt_vertices[task_id],
                  'viz_start_idx': batch_size - group_size}

        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints_outer.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d_outer.detach().item(),
                  'loss_regr_pose': loss_regr_pose_outer.detach().item(),
                  'loss_regr_betas': loss_regr_betas_outer.detach().item(),
                  'loss_shape': loss_shape_outer.detach().item()}
        if cur_step % 10 == 0:
            print(losses)

        return output, losses

    def test(self, epoch):
        self.model.eval()
        self.model_aux.eval()
        for param in self.model_aux.parameters():
            param.requires_grad = False
        # mpjpe, pa_mpjpe, pve = run_evaluation(self.model, self.options.eval_dataset, self.eval_dataset, None,
        #                                  batch_size=self.options.batch_size,
        #                                  shuffle=False,
        #                                  log_freq=50,
        #                                  with_train=True, eval_epoch=epoch, summary_writer=self.summary_writer,
        #                                  bbox_type=self.bbox_type)
        # results = {'mpjpe': mpjpe,
        #            'pa_mpjpe': pa_mpjpe,
        #            'pve': pve}

        mpjpe_3dpw, pa_mpjpe_3dpw, pve_3dpw = run_evaluation(self.model, self.model_aux,
                                                                dataset_name='3dpw',
                                                                dataset=self.eval_dataset_3dpw,
                                                                result_file=None,
                                                                batch_size=self.options.batch_size,
                                                                shuffle=False,
                                                                log_freq=50,
                                                                with_train=True, eval_epoch=epoch,
                                                                summary_writer=self.summary_writer,
                                                                bbox_type=self.bbox_type,
                                                                params=None,
                                                                options=self.options)


        params_test = self.params
        mpjpe_h36m, pa_mpjpe_h36m, pve_h36m = run_evaluation(self.model, self.model_aux,
                                                             'h36m-p2', self.eval_dataset_h36m, None,
                                                             batch_size=self.options.batch_size,
                                                             shuffle=False,
                                                             log_freq=50,
                                                             with_train=True, eval_epoch=epoch,
                                                             summary_writer=self.summary_writer,
                                                             bbox_type=self.bbox_type,
                                                             params=None,
                                                             options=self.options)

        results = {'3dpw1': mpjpe_3dpw,
                    '3dpw2': pa_mpjpe_3dpw,
                    '3dpw3': pve_3dpw,
                    'h36m1': mpjpe_h36m,
                    'h36m2': pa_mpjpe_h36m,
                    'h36m3': pve_h36m}
        
        for param in self.model_aux.parameters():
            param.requires_grad = True

        return results

    def train_summaries(self, input_batch, output, losses):

        # images = input_batch['img']
        # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        #
        # pred_vertices = output['pred_vertices']
        # gt_vertices = output['gt_vertices']
        # # pred_vertices = gt_vertices
        # pred_cam_t = output['pred_cam_full']
        # pred_cam_crop = output['pred_cam_crop']
        # img_names = output['img_name']
        # dataset = output['dataset']
        # focal_length = input_batch['focal_length']
        # center = input_batch['center']
        # rot = input_batch['rot_angle']
        # is_flipped = input_batch['is_flipped']
        # has_smpl = input_batch['has_smpl'].byte()
        # img_h, img_w = input_batch['img_h'], input_batch['img_w']
        # img_to_sum = np.random.choice(len(img_names), 4, replace=False)
        # viz_start_idx = output['viz_start_idx']
        # num = 0
        # for idx in img_to_sum:
        #     print('focal, w, h', focal_length[viz_start_idx + idx], img_w[viz_start_idx + idx],
        #           img_h[viz_start_idx + idx])
        #     crop_renderer = Renderer(focal_length=5000,
        #                              img_res=[self.crop_w, self.crop_h], faces=self.smpl.faces)
        #     renderer = Renderer(focal_length=focal_length[viz_start_idx + idx],
        #                         img_res=[img_w[viz_start_idx + idx], img_h[viz_start_idx + idx]], faces=self.smpl.faces)
        #     rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
        #     if dataset[idx] == '3dpw':
        #         rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1] // 2, rgb_img.shape[0] // 2))
        #         print('Resize 3dpw to half size!')
        #     if is_flipped[viz_start_idx + idx]:
        #         rgb_img = flip_img(rgb_img)
        #     MAR = cv2.getRotationMatrix2D((int(center[viz_start_idx + idx][0]), int(center[viz_start_idx + idx][1])),
        #                                   int(rot[viz_start_idx + idx]), 1.0)
        #     rotated_img = np.transpose(
        #         cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[viz_start_idx + idx]), int(img_h[viz_start_idx + idx]))),
        #         (2, 0, 1)) / 255.0
        #     image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
        #     images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0),
        #                                         pred_cam_t[idx].unsqueeze(0),
        #                                         image_full)
        #     images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0),
        #                                                   gt_vertices[idx].unsqueeze(0),
        #                                                   pred_cam_crop[idx].unsqueeze(0),
        #                                                   images[viz_start_idx + idx:viz_start_idx + idx + 1])
        #     self.summary_writer.add_image('pred_mesh_{}_{}'.format(dataset[idx], num), images_pred, self.step_count)
        #     self.summary_writer.add_image('pred_crop_mesh_{}_{}'.format(dataset[idx], num), images_crop_pred,
        #                                   self.step_count)
        #     num += 1

        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
        if self.use_pseudo:
            err_s = output['err_s']
            self.summary_writer.add_scalar('S', err_s.item(), self.step_count)
        # self.summary_writer.flush()