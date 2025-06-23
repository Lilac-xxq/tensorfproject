import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

# === 轨迹生成相关函数 ===
def normalize(v):
    return v / np.linalg.norm(v)

def average_poses(poses):
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_ = poses[..., 1].mean(0)
    x = normalize(np.cross(z, y_))
    y = np.cross(x, z)
    pose_avg = np.stack([x, y, z, center], 1)
    return pose_avg

def center_poses(poses):
    pose_avg = average_poses(poses)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0,0,0,1]), (poses.shape[0],1,1))
    poses_homo = np.concatenate([poses, last_row], 1)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo
    poses_centered = poses_centered[:, :3]
    return poses_centered, pose_avg_homo

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    dt = 0.75
    close_depth, inf_depth = np.min(near_fars) * 0.9, np.max(near_fars) * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    zdelta = np.min(near_fars) * .2
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)
# === 轨迹生成函数结束 ===

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


class LLFFJsonDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8):
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.hold_every = hold_every
        self.define_transforms()

        # 坐标系变换（如有需要）
        self.blender2opencv = np.eye(4)  # 如果你的json本身就是opencv坐标系

        self.read_meta()
        self.white_bg = True

        # === 中心化/尺度归一化/自动bbox ===
        poses = np.array(self.poses).astype(np.float32)  # (N, 4, 4), float32
        poses_3x4 = poses[:, :3, :]
        poses_centered, pose_avg_homo = center_poses(poses_3x4)
        centers = poses_centered[..., 3]
        # 用所有相机中心到原点的最小距离归一化
        scale_factor = np.min(np.linalg.norm(centers, axis=1))
        poses_centered[..., 3] /= scale_factor
        self.poses = poses_centered  # (N, 3, 4), float32

        # 自动设置scene_bbox
        self.scene_bbox = torch.tensor([
            [centers[:,0].min(), centers[:,1].min(), centers[:,2].min()],
            [centers[:,0].max(), centers[:,1].max(), centers[:,2].max()]
        ], dtype=torch.float32)

        # 自动设置near_far
        dists = np.linalg.norm(centers, axis=1) / scale_factor
        self.near_far = [dists.min() * 0.9, dists.max() * 1.1]

        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # 生成与训练集分布一致的螺旋轨迹
        self.render_path = torch.from_numpy(get_spiral(self.poses, self.near_far, N_views=120).astype(np.float32)).float()

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms.json"), 'r') as f:
            self.meta = json.load(f)
        w, h = int(self.meta['w']/self.downsample), int(self.meta['h']/self.downsample)
        self.img_wh = [w, h]
        self.focal = 0.5 * w / np.tan(0.5 * self.meta['camera_angle_x'])
        self.focal *= self.img_wh[0] / self.meta['w']
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        i_test = np.arange(0, len(self.meta['frames']), self.hold_every)
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.meta['frames']))) - set(i_test))
        self.all_rays = []
        self.all_rgbs = []
        self.poses = []
        for i in img_list:
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses.append(pose)
            c2w = torch.FloatTensor(pose)
            image_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            img = Image.open(image_path).convert('RGB')
            try:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            except Exception:
                img = img.resize(self.img_wh)
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)
            self.all_rgbs.append(img)
            rays_o, rays_d = get_rays(self.directions, c2w)
            self.all_rays.append(torch.cat([rays_o, rays_d], 1))
        self.poses = np.array(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}
        return sample 