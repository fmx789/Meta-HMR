import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import numpy as np
import pytorch3d
import pytorch3d.renderer
import torch
from scipy.spatial.transform import Rotation
import cv2

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

    def visualize_tb(self, pred_vertices, gt_vertices, camera_translation, images, base_color=(0.99, 0.83, 0.5, 1.0), amb_color=[0., 0., 0.], grid=True, blank_bg=False, side=None):
        pred_vertices = pred_vertices.cpu().numpy()
        gt_vertices = gt_vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        print('in render',images.shape)
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        if blank_bg:
            # images_np = np.ones_like(images_np)
            images_np = np.zeros_like(images_np)
        rend_imgs = []
        for i in range(pred_vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(pred_vertices[i], camera_translation[i], images_np[i], base_color=base_color, amb_color=amb_color, aroundy=side), (2,0,1))).float()
            # gt_img = torch.from_numpy(np.transpose(self.__call__(gt_vertices[i], camera_translation[i], images_np[i], color=color), (2,0,1))).float()
            if not grid:
                return rend_img
            
            gt_img = images[i]
            rend_imgs.append(gt_img)
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, base_color=(0.99, 0.83, 0.5, 1.0), amb_color=[0., 0., 0.], aroundy=None):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=base_color,
            alphaCutoff=0.5)
            # baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.
        if aroundy is not None:
            center = vertices.mean(axis=0)
            rot_vertices = np.dot((vertices - center), aroundy) + center
            vertices = rot_vertices

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=amb_color)
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1., 1., 1.], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([-1, -1, -2])
        scene.add(light, pose=light_pose)

        
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        # print(valid_mask.shape)
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
    
class Renderer_Pytorch3d:
    def __init__(self, focal_length=5000, img_res=[224, 224], faces=None):
        # print(img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.height = img_res[1]
        self.width = img_res[0]
        self.faces = faces.unsqueeze(0)

    def render_mesh(self, vertices, translation, images, base_color=(0.99, 0.83, 0.5), grid=True, blank_bg=False, side=None, device=None):
        ''' Render the mesh under camera coordinates
        vertices: (N_v, 3), vertices of mesh
        faces: (N_f, 3), faces of mesh
        translation: (3, ), translations of mesh or camera
        focal_length: float, focal length of camera
        height: int, height of image
        width: int, width of image
        device: "cpu"/"cuda:0", device of torch
        :return: the rgba rendered image
        '''
        if device is None:
            device = vertices.device

        bs = vertices.shape[0]

        # add the translation
        vertices = vertices + translation[:, None, :]

        # upside down the mesh
        # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
        rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
        rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
        # self.faces = self.faces.expand(bs, *self.faces.shape).to(device)
        self.faces = self.faces.to(device)
        images = images.cpu().numpy()

        vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)

        if blank_bg:
            aa_factor = 1  # anti-aliasing factor
            input_img = np.ones_like(images) * 255.
        else:
            aa_factor = 1  # anti-aliasing factor
            input_img = images

        if side is not None:
            side = torch.from_numpy(side).to(torch.float32).to(device).unsqueeze(0)
            center = vertices.mean(axis=1)
            rot_vertices = torch.matmul((vertices - center), side) + center
            vertices = rot_vertices

        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
        textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
        mesh = pytorch3d.structures.Meshes(verts=vertices, faces=self.faces, textures=textures)

        # Initialize a camera.
        cameras = pytorch3d.renderer.PerspectiveCameras(
            focal_length=((2 * self.focal_length / min(self.height, self.width), 2 * self.focal_length / min(self.height, self.width)),),
            device=device,
        )

        # Define the settings for rasterization and shading.
        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=(self.height*aa_factor, self.width*aa_factor),   # (H*aa_factor, W*aa_factor)
            # image_size=height,   # (H, W)
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )

        # Define the material
        materials = pytorch3d.renderer.Materials(
            ambient_color=(base_color[:3],),
            diffuse_color=((1, 1, 1),),
            specular_color=((1, 1, 1),),
            shininess=64,
            device=device
        )

        # Place a directional light in front of the object.
        lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

        # Create a phong renderer by composing a rasterizer and a shader.
        renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                materials=materials
            )
        )

        # Do rendering
        if blank_bg:
            imgs = renderer(mesh)
            imgs = imgs.permute(0,3,1,2) # NHWC->NCHW
            imgs = F.avg_pool2d(imgs,kernel_size=aa_factor,stride=aa_factor)
            imgs = imgs.permute(0,2,3,1).cpu().numpy() * 255. # NHWC
        else:
            imgs = renderer(mesh).cpu().numpy() * 255.


        color_batch = imgs
        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch)

        color = image_vis_batch
        valid_mask = valid_mask_batch

        # image_vis = input_img
        alpha = 0.9
        # image_vis = ((1 - alpha) * input_img + alpha * input_img)
        image_vis = alpha * color[..., :3] * valid_mask + (
            1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

        # image_vis = image_vis.astype(np.uint8)
        # image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        # res_path = os.path.join(opt.out_dir, basename)
        # cv2.imwrite(res_path, image_vis)

        return image_vis[0, ..., :3]
