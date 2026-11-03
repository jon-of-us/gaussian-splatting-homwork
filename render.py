#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
from utils.sh_utils import eval_sh
from utils.general_utils import getWorld2View, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy,image_width, image_height):
        super(Camera, self).__init__()
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        
        self.data_device = torch.device("cuda")

        self.image_width = image_width
        self.image_height = image_height
        
        self.zfar = 100.0
        self.znear = 0.01

        self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Precompute 3d covariance.
    scales = None
    rotations = None
    cov3D_precomp = pc.get_covariance(scaling_modifier)

    # Precompute colors.
    shs = None
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

        
def render_sets(args):
    # load camera configs from assets/cameras/*.npy
    def load_cameras():
        cameras = []
        for camera_file in sorted(os.listdir("assets/cameras")):
            camera_config=np.load(os.path.join("assets/cameras",camera_file),allow_pickle=True).item()
            camera=Camera(camera_config["R"], camera_config["T"], camera_config["FoVx"], camera_config["FoVy"], camera_config["image_width"], camera_config["image_height"])
            cameras.append(camera)
        return cameras
    
    cameras=load_cameras()
    
    with torch.no_grad():
        gaussians = GaussianModel(args.sh_degree)
        gaussians.load_ply(args.model_path)
        
        ### Begin Code 3.2 ###
        points_before = gaussians.get_xyz.shape[0]
        gaussians.prune_points(args.min_opacity)
        points_after = gaussians.get_xyz.shape[0]
        if points_after == 0:
            raise ValueError(
                f"Opacity threshold {args.min_opacity} removed all Gaussians."
            )
        pruning_ratio = 1.0 - points_after / points_before
        print(f"Number of Gaussians before pruning: {points_before}")
        print(f"Number of Gaussians after pruning: {points_after}")
        print(f"Pruning ratio at opacity threshold {args.min_opacity}: {pruning_ratio:.2%}")
        ### End code 3.2 ###
        
        bg_color = [0, 0, 0] # Black background for our scene
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_path="renders"
        makedirs(render_path, exist_ok=True)

        for idx, view in enumerate(tqdm(cameras, desc="Rendering progress")):
            rendering = render(view, gaussians, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path","-m", default="assets/gs_cloud.ply",type=str)
    parser.add_argument("--sh_degree", default=1, type=int, help="active SH degree")
    parser.add_argument("--min_opacity", default=0.0, type=float, help="prune Gaussians below this opacity")
    args=parser.parse_args()
    render_sets(args)
