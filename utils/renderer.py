import colorsys
import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self, focal_length=5000, img_res=[224, 224], faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                                   viewport_height=img_res[1],
                                                   point_size=1.0)
        # print(img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces

    def visualize_tb(self, pred_vertices, gt_vertices, camera_translation, images, grid=True, color=[1., 1., 1.],
                     baseColorFactor=colorsys.hsv_to_rgb(0.5418,1.,1.), blank_bg=False, mesh_filename=None):
        pred_vertices = pred_vertices.cpu().numpy()
        gt_vertices = gt_vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        print('in render', images.shape)
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        if blank_bg:
            images_np = np.ones_like(images_np)
        rend_imgs = []
        for i in range(pred_vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(
                self.__call__(pred_vertices[i], camera_translation[i], images_np[i], color=color,
                              baseColorFactor=baseColorFactor, mesh_filename=mesh_filename), (2, 0, 1))).float()
            if mesh_filename is not None:
                mesh_filename = mesh_filename[:-len(mesh_filename.split('_')[-1])] + 'gt_dataset.obj'
                gt_img = torch.from_numpy(np.transpose(
                    self.__call__(gt_vertices[i], camera_translation[i], images_np[i], color=color,
                                  baseColorFactor=baseColorFactor, mesh_filename=mesh_filename), (2, 0, 1))).float()
            if not grid:
                return rend_img

            gt_img = images[i]
            rend_imgs.append(gt_img)
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, color, baseColorFactor, mesh_filename=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        if mesh_filename is not None:
            mesh.export(mesh_filename)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene = pyrender.Scene(ambient_light=(0., 0., 0.))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        # light = pyrender.DirectionalLight(color=color, intensity=1)
        # light_pose = np.eye(4)
        #
        # light_pose[:3, 3] = np.array([0, -1, 1])
        # scene.add(light, pose=light_pose)
        #
        # light_pose[:3, 3] = np.array([0, 1, 1])
        # scene.add(light, pose=light_pose)
        #
        # light_pose[:3, 3] = np.array([1, 1, 2])
        # scene.add(light, pose=light_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        # print(valid_mask.shape)
        output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image)
        # output_img = (color[:, :, :3] * valid_mask +
        #               (1-valid_mask)*np.ones(color.shape).astype(np.float32)[:, :, :3])
        return output_img
