"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""
from copy import deepcopy
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm
from collections import OrderedDict

import config
import constants
from models import hmr, SMPL
from datasets import BaseDataset
from models.maml import gradient_update_parameters
from models.meta_models import build_model
from utils.geometry import cam_crop2full, perspective_projection, batch_rodrigues
from utils.imutils import uncrop, flip_img
from utils.pose_utils import reconstruction_error
# from utils.part_utils import PartRenderer
from utils.renderer import Renderer
import torch.nn as nn

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='hmr', help='name of exp model')
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--backbone', default='resnet', help='backbone')
parser.add_argument('--dataset', default='h36m-p1',
                    choices=['h36m-p1', 'h36m-p2', '3dpw', 'mpi-inf-3dhp', 'coco'],
                    help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=40, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')
parser.add_argument('--viz', default=False, action='store_true', help='Visualize the mesh result')
parser.add_argument('--use_latent', default=False, action='store_true', help='Use latent encoder')
parser.add_argument('--use_pseudo', default=False, action='store_true', help='Use pseudo labels')
parser.add_argument('--out_num', type=int, default=3, help='Number of ouptput value')
parser.add_argument('--use_extraviews', default=False, action='store_true', help='Use latent encoder')
parser.add_argument('--use_fuse', default=False, action='store_true', help='Use pseudo labels')
parser.add_argument('--bbox_type', default='rect',
                    help='Use square bounding boxes in 224x224 or rectangle ones in 256x192')
parser.add_argument('--rescale_bbx', default=False, action='store_true',
                    help='Use rescaled bbox for consistency data aug and loss')
parser.add_argument('--shift_center', default=False, action='store_true',
                    help='Use shifted center for consistency data aug and loss')
parser.add_argument('--inner_lr', type=float, default=1e-5, help='the inner lr')
parser.add_argument('--test_val_step', type=int, default=14, help='# of inner step')
parser.add_argument('--first_order', action='store_true', help='use the first order gradient')
parser.add_argument('--inner_step', type=int, default=1, help='# of inner step')
parser.add_argument('--no_use_adam', action='store_true', help='not use_adam after 1 inner step')
parser.add_argument('--no_inner_step', action='store_true', help='use_adam after 1 inner step')
parser.add_argument('--after_innerlr', type=float, default=1e-5, help='the inner lr')
parser.add_argument('--shape_loss_weight', default=0.5, type=float, help='Weight of per-vertex loss')
parser.add_argument('--keypoint_loss_weight', default=5., type=float, help='Weight of 2D and 3D keypoint loss')
parser.add_argument('--pose_loss_weight', default=1., type=float, help='Weight of SMPL pose loss')
parser.add_argument('--beta_loss_weight', default=0.001, type=float, help='Weight of SMPL betas loss')
parser.add_argument('--openpose_train_weight', default=0., help='Weight for OpenPose keypoints during training')
parser.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training')
parser.add_argument("--use_openpose2d", default=False, action='store_true', help='Use OpenPose detect 2d for during evaluation')
parser.add_argument("--adapt_val_step", default=False, action='store_true', help='adaptive val step, which will disable original val step')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
criterion_shape = nn.L1Loss().to(device)
# Keypoint (2D and 3D) loss
# No reduction because confidence weighting needs to be applied
criterion_keypoints = nn.MSELoss(reduction='none').to(device)
# Loss for SMPL parameter regression
criterion_regr = nn.MSELoss().to(device)


def run_evaluation(model, model_aux, dataset_name, dataset, result_file,
                   batch_size=50, img_res=224,
                   num_workers=0, shuffle=False, log_freq=50,
                   with_train=False, eval_epoch=None, summary_writer=None, viz=False, bbox_type='square', params=None,
                   options=None):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)
    if model_aux is not None:
        model_aux.to(device)

    print(bbox_type)
    if bbox_type == 'square':
        crop_w = 224.
        crop_h = 224.
        bbox_size = 224.
    elif bbox_type == 'rect':
        crop_w = 192.
        crop_h = 256.
        bbox_size = 256.
    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)

    # renderer = PartRenderer()

    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle = False
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    pve = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == 'coco':
        eval_pose = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14


    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        model_copy = deepcopy(model)
        params_test = OrderedDict(model_copy.meta_named_parameters())

        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'].to(device)
        bbox_info = batch['bbox_info'].to(device)
        curr_batch_size = images.shape[0]
        img_h = batch['img_h'].to(device)
        img_w = batch['img_w'].to(device)
        focal_length = batch['focal_length'].to(device)
        img_names = batch['imgname']
        center = batch['center'].to(device)
        scale = batch['scale'].to(device)
        sc = batch['scale_factor'].to(device)
        rot = batch['rot_angle'].to(device)
        gt_keypoints_2d = batch['keypoints'].to(device)
        is_flipped = batch['is_flipped'].to(device)
        crop_trans = batch['crop_trans'].to(device)
        has_smpl = batch['has_smpl'].byte().to(device)  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = batch['has_pose_3d'].byte().to(device)

        inner_lr = options.inner_lr

        if not options.no_use_adam:
            optimAdam = torch.optim.Adam(model_copy.parameters(), lr=options.after_innerlr, betas=[0.9, 0.999])
        img_h, img_w = img_h.view(-1, 1), img_w.view(-1, 1)
        full_img_shape = torch.hstack((img_h, img_w))
        camera_center = torch.zeros(curr_batch_size, 2, device=device)
        for in_step in range(options.test_val_step + 1):
            pred_rotmat_inner, pred_betas_inner, pred_camera_inner = model_copy(images, params=params_test)
            pred_output_inner = smpl_neutral(betas=pred_betas_inner, body_pose=pred_rotmat_inner[:, 1:],
                                             global_orient=pred_rotmat_inner[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices_inner = pred_output_inner.vertices
            pred_joints_inner = pred_output_inner.joints

            pred_cam_t = torch.stack([pred_camera_inner[:, 1],
                                      pred_camera_inner[:, 2],
                                      2 * constants.FOCAL_LENGTH / (bbox_size * pred_camera_inner[:, 0] + 1e-9)],
                                     dim=-1)

            pred_keypoints_2d = perspective_projection(pred_joints_inner,
                                                       rotation=torch.eye(3, device=device).unsqueeze(
                                                           0).expand(curr_batch_size, -1, -1),
                                                       translation=pred_cam_t,
                                                       focal_length=constants.FOCAL_LENGTH,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            gt_keypoints2d_norm = gt_keypoints_2d.clone()
            pred_keypoints_2d[:, :, 0] = pred_keypoints_2d[:, :, 0] / (crop_w / 2.)
            gt_keypoints2d_norm[:, :, 0] = 2. * gt_keypoints2d_norm[:, :, 0] / crop_w - 1.
            pred_keypoints_2d[:, :, 1] = pred_keypoints_2d[:, :, 1] / (crop_h / 2.)
            gt_keypoints2d_norm[:, :, 1] = 2. * gt_keypoints2d_norm[:, :, 1] / crop_h - 1.

            if options.use_openpose2d:
                loss_keypoints = keypoint_loss(pred_keypoints_2d.float(), gt_keypoints2d_norm.float(), openpose_weight=1.,
                                     gt_weight=0.)
            else:
                loss_keypoints = keypoint_loss(pred_keypoints_2d.float(), gt_keypoints2d_norm.float(), openpose_weight=0.,
                                     gt_weight=1.)

            if in_step == 0:
                pred_rotmat_aux, pred_betas_aux, pred_camera_aux = model_aux(images)
                pred_output_aux = smpl_neutral(betas=pred_betas_aux, body_pose=pred_rotmat_aux[:, 1:],
                                               global_orient=pred_rotmat_aux[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices_aux = pred_output_aux.vertices
                pred_joints_aux = pred_output_aux.joints
                loss_regr_pose, loss_regr_betas = smpl_losses(pred_rotmat_inner, pred_betas_inner,
                                                              pred_rotmat_aux, pred_betas_aux,
                                                              has_smpl, use_pseudo=True)
                loss_keypoints_3d = keypoint_3d_loss(pred_joints_inner, pred_joints_aux, has_pose_3d, use_pseudo=True)
                loss_shape = shape_loss(pred_vertices_inner, pred_vertices_aux, has_smpl)

                loss = options.shape_loss_weight * loss_shape + \
                       options.keypoint_loss_weight * loss_keypoints + \
                       options.keypoint_loss_weight * loss_keypoints_3d + \
                       options.pose_loss_weight * loss_regr_pose + options.beta_loss_weight * loss_regr_betas + \
                       ((torch.exp(-pred_camera_inner[:, 0] * 10)) ** 2).mean()

            if in_step > 0 or options.no_inner_step:
                if options.adapt_val_step:
                    loss_prev = loss
                loss = loss_keypoints
            if in_step > 0:
                inner_lr = options.after_innerlr

            if (in_step == 0 or options.no_use_adam) and not options.no_inner_step:
                model_copy.zero_grad()
                params_test = gradient_update_parameters(model_copy, loss, params=params_test,
                                                         step_size=inner_lr, first_order=True)
                write_params(model_copy, params_test)  # for multi-step optim
            else:
                if in_step == 0:
                    write_params(model_copy, params_test)
                optimAdam.zero_grad()
                loss.backward()
                optimAdam.step()
                params_test = OrderedDict(model_copy.named_parameters())
                if options.adapt_val_step:
                    # early stop
                    if abs(loss_prev - loss) < 2e-5:
                        break

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model_copy(images, params=params_test)


            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                       global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices


        # if viz:
        #     pred_cam_crop = torch.stack([pred_camera[:, 1],
        #                                  pred_camera[:, 2],
        #                                  2 * 5000 / (bbox_size * pred_camera[:, 0] + 1e-9)],
        #                                 dim=-1)
        #     full_img_shape = torch.stack((img_h, img_w), -1)
        #     pred_cam_full = cam_crop2full(pred_camera, center, scale, full_img_shape,
        #                                   focal_length).to(torch.float32)
        #     images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        #     images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        #     # print(images.shape)
        #     num = 0
        #     for idx in tqdm(np.random.choice(len(img_names), 4, replace=False)):
        #         name = img_names[idx].split('/')[-1].split('.')[0]
        #         print(name)
        #         renderer = Renderer(focal_length=focal_length[idx],
        #                             img_res=[img_w[idx], img_h[idx]], faces=smpl_neutral.faces)
        #         # print(crop_w, crop_h)
        #         crop_renderer = Renderer(focal_length=5000,
        #                                  img_res=[crop_w, crop_h], faces=smpl_neutral.faces)
        #         image_mat = np.ascontiguousarray((images[idx] * 255.0).permute(1, 2, 0).cpu().detach().numpy())
        #         for kp in gt_keypoints_2d[idx, 25:]:
        #             cv2.circle(image_mat, (int(kp[0]), int(kp[1])), radius=3, color=(0, 0, 255), thickness=-1)
        #
        #         rgb_img = cv2.imread(img_names[idx])[:, :, ::-1].copy().astype(np.float32)
        #         # rgb_img = np.full((int(img_h[idx]), int(img_w[idx]), 3), fill_value=255., dtype=np.float32)
        #         if is_flipped[idx]:
        #             rgb_img = flip_img(rgb_img)
        #         if dataset_name == '3dpw':
        #             rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1] // 2, rgb_img.shape[0] // 2))
        #         MAR = cv2.getRotationMatrix2D((int(center[idx][0]), int(center[idx][1])), int(rot[idx]), 1.0)
        #         rotated_img = np.transpose(cv2.warpAffine(rgb_img.copy(), MAR, (int(img_w[idx]), int(img_h[idx]))),
        #                                    (2, 0, 1)) / 255.0
        #         image_full = torch.from_numpy(rotated_img).to(images.device).unsqueeze(0)
        #         images_pred = renderer.visualize_tb(pred_vertices[idx].unsqueeze(0), gt_vertices[idx].unsqueeze(0),
        #                                             pred_cam_full[idx].unsqueeze(0), image_full, grid=False)
        #         images_crop_pred = crop_renderer.visualize_tb(pred_vertices[idx].unsqueeze(0),
        #                                                       gt_vertices[idx].unsqueeze(0),
        #                                                       pred_cam_crop[idx].unsqueeze(0), images[idx:idx + 1])
        #         cv2.imwrite('eval_result/img_viz/{}_img_visualized.jpg'.format(num),
        #                     np.ascontiguousarray(images_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
        #         cv2.imwrite('eval_result/img_viz/{}_img_crop_visualized.jpg'.format(num),
        #                     np.ascontiguousarray(images_crop_pred * 255.0).transpose(1, 2, 0)[:, :, ::-1])
        #         cv2.imwrite('eval_result/img_viz/{}_img_crop_origin.jpg'.format(num), image_mat[:, :, ::-1])
        #         num = num + 1


        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            # if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
            if 'mpi-inf' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                # print("Using smpl joints as gt 3d joints!", has_smpl[0])
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:],
                                                 betas=gt_betas).vertices
                torch.set_printoptions(threshold=10000)
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(
                dim=-1).cpu().numpy()
            pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error
            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :,
                :] = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),
                                           reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error

            # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print('PVE: ' + str(1000 * pve[:step * batch_size].mean()))
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print('PVE: ' + str(1000 * pve.mean()))
        print()

    if with_train:
        summary_writer.add_scalar(dataset_name + '/' + 'MPJPE', 1000 * mpjpe.mean(), eval_epoch)
        summary_writer.add_scalar(dataset_name + '/' + 'PA-MPJPE', 1000 * recon_err.mean(), eval_epoch)
        summary_writer.add_scalar(dataset_name + '/' + 'PVE', 1000 * pve.mean(), eval_epoch)
        return 1000 * mpjpe.mean(), 1000 * recon_err.mean(), 1000 * pve.mean()


def compute_keypoints2d_loss_cliff(
        pred_keypoints3d: torch.Tensor,
        pred_cam: torch.Tensor,
        gt_keypoints2d: torch.Tensor,
        camera_center: torch.Tensor,
        focal_length: torch.Tensor,
        crop_trans,
        crop_h,
        crop_w,
        use_openpose2d):
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
        (pred_keypoints2d_full, torch.ones(batch_size, 49, 1).to(device)),
        dim=2)
    # trans @ pred_keypoints2d2
    pred_keypoints2d_bbox = torch.einsum('bij,bkj->bki', crop_trans,
                                         pred_keypoints2d)

    pred_keypoints2d_bbox[:, :, 0] = 2. * pred_keypoints2d_bbox[:, :, 0] / crop_w - 1.
    gt_keypoints2d[:, :, 0] = 2. * gt_keypoints2d[:, :, 0] / crop_w - 1.
    pred_keypoints2d_bbox[:, :, 1] = 2. * pred_keypoints2d_bbox[:, :, 1] / crop_h - 1.
    gt_keypoints2d[:, :, 1] = 2. * gt_keypoints2d[:, :, 1] / crop_h - 1.

    if use_openpose2d:
        loss = keypoint_loss(pred_keypoints2d_bbox.float(), gt_keypoints2d.float(), openpose_weight=1., gt_weight=0.)
    else:
        loss = keypoint_loss(pred_keypoints2d_bbox.float(), gt_keypoints2d.float(), openpose_weight=0., gt_weight=1.)

    return loss


def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, openpose_weight=0., gt_weight=1.):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """

    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    # torch.set_printoptions(threshold=100000)
    # print(conf)
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss


def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, use_pseudo=False):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    if use_pseudo:
        conf = torch.ones_like(gt_keypoints_3d[:, :, -1].unsqueeze(-1)[:, 25:, :], device=device)
        gt_keypoints_3d = gt_keypoints_3d.clone()[:, 25:, :]
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
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)


def shape_loss(pred_vertices, gt_vertices, has_smpl):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]

    if len(gt_vertices_with_shape) > 0:
        return criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)


def smpl_losses(pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, use_pseudo=False):
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    if use_pseudo:
        gt_rotmat_valid = gt_pose[has_smpl == 1]
    else:
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = criterion_regr(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(device)
    return loss_regr_pose, loss_regr_betas


def write_params(model, param_dict):
    for name, param in model.meta_named_parameters():
        param.data.copy_(param_dict[name])


if __name__ == '__main__':
    seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()
    result_file = os.path.join('logs', 'params.txt')
    logger = open(os.path.join(result_file), mode='w+', encoding='utf-8')
    model = build_model(config.SMPL_MEAN_PARAMS, pretrained=False, backbone=args.backbone, name='hmr_meta',
                        bbox_type=args.bbox_type)
    model_aux = build_model(config.SMPL_MEAN_PARAMS, pretrained=False, backbone=args.backbone, name='hmr_aux',
                            bbox_type=args.bbox_type)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=True)
    model_params = OrderedDict(model.meta_named_parameters())
    model.eval()
    model_aux.load_state_dict(checkpoint['model_aux'], strict=True)
    # set aux net grad false
    for param in model_aux.parameters():
        param.requires_grad = False
    model_aux.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(args, args.dataset, is_train=False, bbox_type=args.bbox_type)
    # Run evaluation
    run_evaluation(model, model_aux, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq,
                   viz=args.viz,
                   bbox_type=args.bbox_type,
                   options=args,
                   )
