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
import os
from os import makedirs
import torch
import numpy as np

import subprocess

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from omegaconf import DictConfig

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, show_level, ape_code):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_path, exist_ok=True)
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(gts_path, exist_ok=True)
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    t_list = []
    per_view_dict = {}
    per_view_level_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()

        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
        
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()  
        per_view_dict['{0:05d}'.format(idx)+".png"] = visible_count.item()

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if show_level:
            for cur_level in range(gaussians.levels):
                gaussians.set_anchor_mask_perlevel(view.camera_center, view.resolution_scale, cur_level)
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, ape_code=ape_code)
                
                rendering = render_pkg["render"]
                visible_count = render_pkg["visibility_filter"].sum()
                
                torchvision.utils.save_image(rendering, os.path.join(render_level_path, '{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"))
                per_view_level_dict['{0:05d}_LOD{1:d}'.format(idx, cur_level) + ".png"] = visible_count.item()

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True) 
    if show_level:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count_level.json"), 'w') as fp:
            json.dump(per_view_level_dict, fp, indent=True)     

def gen_traj(viewpoint_cameras, n_frames=480):
    import copy
    def normalize(x: np.ndarray) -> np.ndarray:
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = normalize(lookdir)
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m
    def focus_point_fn(poses: np.ndarray) -> np.ndarray:
        """Calculate nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt
    def pad_poses(p: np.ndarray) -> np.ndarray:
        """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
        bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
        return np.concatenate([p[..., :3, :4], bottom], axis=-2)
    def unpad_poses(p: np.ndarray) -> np.ndarray:
        """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
        return p[..., :3, :4]
    def generate_ellipse_path(poses: np.ndarray,
                                n_frames: int = 120,
                                const_speed: bool = True,
                                z_variation: float = 0.,
                                z_phase: float = 0.) -> np.ndarray:
        """Generate an elliptical render path based on the given poses."""
        # Calculate the focal point for the path (cameras point toward this).
        center = focus_point_fn(poses)
        # Path height sits at z=0 (in middle of zero-mean capture pattern).
        offset = np.array([center[0], center[1], 0])

        # Calculate scaling for ellipse axes based on input camera positions.
        sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
        # Use ellipse that is symmetric about the focal point in xy.
        low = -sc + offset
        high = sc + offset
        # Optional height variation need not be symmetric
        z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
        z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

        def get_positions(theta):
            # Interpolate between bounds with trig functions to get ellipse in x-y.
            # Optionally also interpolate in z to change camera height along path.
            return np.stack([
                low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
                low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
                z_variation * (z_low[2] + (z_high - z_low)[2] *
                            (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
            ], -1)

        theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
        positions = get_positions(theta)

        #if const_speed:

        # # Resample theta angles so that the velocity is closer to constant.
        # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
        # positions = get_positions(theta)

        # Throw away duplicated last position.
        positions = positions[:-1]

        # Set path's up vector to axis closest to average of input pose up vectors.
        avg_up = poses[:, :3, 1].mean(0)
        avg_up = avg_up / np.linalg.norm(avg_up)
        ind_up = np.argmax(np.abs(avg_up))
        up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

        return np.stack([viewmatrix(p - center, up, p) for p in positions])
    def transform_poses_pca(poses: np.ndarray):
        """Transforms poses so principal components lie on XYZ axes.

        Args:
            poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

        Returns:
            A tuple (poses, transform), with the transformed poses and the applied
            camera_to_world transforms.
        """
        t = poses[:, :3, 3]
        t_mean = t.mean(axis=0)
        t = t - t_mean

        eigval, eigvec = np.linalg.eig(t.T @ t)
        # Sort eigenvectors in order of largest to smallest eigenvalue.
        inds = np.argsort(eigval)[::-1]
        eigvec = eigvec[:, inds]
        rot = eigvec.T
        if np.linalg.det(rot) < 0:
            rot = np.diag(np.array([1, 1, -1])) @ rot

        transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
        poses_recentered = unpad_poses(transform @ pad_poses(poses))
        transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

        # Flip coordinate system if z component of y-axis is negative
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
            transform = np.diag(np.array([1, -1, -1, 1])) @ transform

        return poses_recentered, transform
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        device = cam.world_view_transform.device
        cam.camera_center = cam.world_view_transform.cpu().inverse()[3, :3].to(device)
        traj.append(cam)

    return traj

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, new_traj : bool, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()
        gaussians.plot_levels()
        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if new_traj:
            cameras = gen_traj(scene.getTrainCameras())
            render_set(dataset.model_path, "gen", scene.loaded_iter, cameras, gaussians, pipeline, background, show_level, ape_code)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level, ape_code)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level, ape_code)



def get_combined_args(parser : ArgumentParser, cfg:DictConfig):
    import sys
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    if "model_path" not in cmdlne_string:
        cmdlne_string.extend(["--model_path", cfg.model_path])
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def main(cfg: DictConfig):
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--gen_traj", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser, cfg)

    # merge hydra into args
    for key, value in vars(args).items():
        setattr(args, key, getattr(cfg, key, value))

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.gen_traj, args.skip_train, args.skip_test, args.show_level, args.ape)
    
