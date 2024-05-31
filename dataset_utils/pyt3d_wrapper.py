"""
a simple wrapper for pytorch3d rendering
"""

import numpy as np
import torch
import pytorch3d
import time
from copy import deepcopy
# Data structures and functions for rendering
from pytorch3d.renderer import (
    PointLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene


SMPL_OBJ_COLOR_LIST = [
    [135 / 255.0, 235 / 255.0, 156 / 255.0],  # person1
    [117 / 255.0, 157 / 255.0, 231 / 255.0],  # person2
    [224 / 255.0, 193 / 255.0, 240 / 255.0],  # object
]


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"
    def __init__(self, image_size=(1200, 900),
                 faces_per_pixel=1,
                 device='cuda:0',
                 blur_radius=0, lights=None,
                 materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel=faces_per_pixel
        self.max_faces_per_bin=max_faces_per_bin # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights=lights if lights is not None else AmbientLights(
            ambient_color=((0.5, 0.5, 0.5),), device=device
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size[::-1],  # input: (h, w)
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin
        )
        shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
            materials=self.materials)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
                shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device='cuda:0', colors=SMPL_OBJ_COLOR_LIST, use_fixed_cameras=False, eyes=None, intrin=None, extrin=None, lights=None):
        if lights is None:
            lights=PointLights(device=device, location=[[0.0, 2, 0.0]])
        self.renderer = MeshRendererWrapper(image_size, device=device, lights=lights)
        self.use_fixed_cameras = use_fixed_cameras
        self.intrin = intrin
        self.image_size = image_size
        if use_fixed_cameras:
            focal_length = torch.tensor((-intrin[0, 0], -intrin[1, 1]), dtype=torch.float32).unsqueeze(0)
            cam_center = torch.tensor((intrin[0, 2], intrin[1, 2]), dtype=torch.float32).unsqueeze(0)
            print(cam_center)
            self.R = torch.from_numpy(extrin[:3, :3].T).unsqueeze(0)
            self.T = torch.from_numpy(extrin[:3, 3]).unsqueeze(0)
            pyt3d_version = pytorch3d.__version__
            if pyt3d_version >= '0.6.0':
                self.cameras = [PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((image_size[1], image_size[0]),), device=device, R=self.R, T=self.T, in_ndc=False)]
            else:
                self.cameras = [PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((image_size[1], image_size[0]),), device=device, R=self.R, T=self.T)]
        else:
            self.cameras = self.get_surround_cameras(self, intrin, image_size, eyes=eyes, n_poses=len(eyes) if not eyes is None else 1, device=device)
        self.colors = deepcopy(colors)
        self.device = device
    
    @staticmethod
    def get_surround_cameras(self, intrin, image_size, radius=3.0, eyes=None, n_poses=20, up=(0.0, 1.0, 0.0), device='cuda:0'):
        fx, fy = (-self.intrin[0, 0], -self.intrin[1, 1])  # focal length
        cx, cy = (self.intrin[0, 2], self.intrin[1, 2])  # camera centers
        color_w, color_h = self.image_size  # kinect color image size
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)

        cameras = []
        if eyes is None:
            eyes = []
            for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
                if np.abs(up[1]) > 0:
                    eye = [np.cos(theta + np.pi / 2) * radius, 0.0, -np.sin(theta + np.pi / 2) * radius]
                else:
                    eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 0.0]
                print(eye)
                eyes.append(eye)
        
        for eye in eyes:
            self.R, self.T = look_at_view_transform(
                eye=(eye,),
                at=([0.0, 0.0, 0.0],),
                up=(up,),
            )
            self.extrin = np.eye(4)
            self.extrin[:3, :3] = self.R.squeeze().cpu().numpy().T
            self.extrin[:3, 3] = self.T.cpu().numpy()
            self.intrin = np.eye(3)
            self.intrin[0, 0] = -fx
            self.intrin[1, 1] = -fy
            self.intrin[0, 2] = cx
            self.intrin[1, 2] = cy
            self.intrin[2, 2] = 1.0
            pyt3d_version = pytorch3d.__version__
            if pyt3d_version >= '0.6.0':
                cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                                        image_size=((color_h, color_w),),
                                        device=device,
                                        R=self.R, T=self.T, in_ndc=False)
            else:
                cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                                        image_size=((color_h, color_w),),
                                        device=device,
                                        R=self.R, T=self.T)
            cameras.append(cam)
        return cameras

    def render_meshes(self, meshes, specified_vertices=None):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        colors = deepcopy(self.colors)
        pyt3d_mesh = self.prepare_render(meshes, colors, specified_vertices)
        rends = []
        for cam in self.cameras:
            img = self.renderer.render(pyt3d_mesh, cam)
            rends.append(img)
        return rends

    def prepare_render(self, meshes, colors, specified_vertices, static_contact_color=[0.9, 0.2, 0.2]):
        py3d_meshes = []
        if specified_vertices is None:
            specified_vertices = [None] * len(meshes)
        for mesh, sp_vs, color in zip(meshes, specified_vertices, colors):
            vc = np.zeros_like(mesh.vertices)
            vc[:, :] = color
            if not sp_vs is None:  # render contact
                vc[sp_vs] = static_contact_color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes([torch.from_numpy(mesh.vertices).float().to(self.device)], [torch.from_numpy(mesh.faces.astype(int)).long().to(self.device)],
                               text)
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined
