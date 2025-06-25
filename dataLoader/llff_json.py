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

def get_spiral(c2ws_all, near_fars, rads_scale=1, N_views=120): # rads_scale参数控制新路径的半径，之前训练有花影版本为1
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

def get_spiral_from_centers(camera_centers, up=None, rads_scale=0.8, N_views=120):
    """
    直接用原始相机中心（蓝点）生成螺旋路径。
    camera_centers: (N, 3) array
    up: (3,) array, optional, 默认为z轴向上
    rads_scale: 半径缩放因子
    N_views: 生成视角数
    """
    center_mean = np.mean(camera_centers, axis=0)
    # 主平面法向量
    if up is None:
        up = np.array([0, 0, 1])
    # 半径（用90分位数而不是最大值）
    radius = np.percentile(np.linalg.norm(camera_centers - center_mean, axis=1), 90) * rads_scale
    # 生成圆环上的点
    angles = np.linspace(0, 2*np.pi, N_views)
    circle_points = np.stack([
        center_mean[0] + radius * np.cos(angles),
        center_mean[1] + radius * np.sin(angles),
        np.full_like(angles, center_mean[2])
    ], axis=1)
    # 生成c2w矩阵（简单实现：z轴朝向物体中心，y轴up）
    render_poses = []
    for c in circle_points:
        z = normalize(center_mean - c)
        x = normalize(np.cross(up, z))
        y = np.cross(z, x)
        c2w = np.eye(4)
        c2w[:3, 0] = x
        c2w[:3, 1] = y
        c2w[:3, 2] = z
        c2w[:3, 3] = c
        render_poses.append(c2w[:3])
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
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, use_blue_path=False, blue_rads_scale=1.0):
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.hold_every = hold_every
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = True

        # === 中心化/尺度归一化/自动bbox ===
        poses = np.array(self.poses).astype(np.float32)  # (N, 4, 4), float32
        poses_3x4 = poses[:, :3, :]
        poses_centered, pose_avg_homo = center_poses(poses_3x4)
        poses_centered = poses_centered.astype(np.float32)  # float32
        centers = poses_centered[..., 3]
        scale_factor = np.max(np.linalg.norm(centers, axis=1))
        poses_centered[..., 3] /= scale_factor
        self.poses = poses_centered  # (N, 3, 4), float32
        # 自动设置scene_bbox
        # 绿点aabb（中心化）保留但注释掉
        # self.scene_bbox = torch.tensor([
        #     [centers[:,0].min(), centers[:,1].min(), centers[:,2].min()],
        #     [centers[:,0].max(), centers[:,1].max(), centers[:,2].max()]
        # ], dtype=torch.float32)
        # 蓝点aabb（原始相机中心）保留但注释掉
        # camera_centers = np.array([np.array(f['transform_matrix'])[:3, 3] for f in self.meta['frames']])
        # self.scene_bbox = torch.tensor([
        #     [camera_centers[:,0].min(), camera_centers[:,1].min(), camera_centers[:,2].min()],
        #     [camera_centers[:,0].max(), camera_centers[:,1].max(), camera_centers[:,2].max()]
        # ], dtype=torch.float32)
        # 只用蓝点aabb并扩展z轴30%
        camera_centers = np.array([np.array(f['transform_matrix'])[:3, 3] for f in self.meta['frames']])
        self.scene_bbox = torch.tensor([
            [camera_centers[:,0].min(), camera_centers[:,1].min(), camera_centers[:,2].min()],
            [camera_centers[:,0].max(), camera_centers[:,1].max(), camera_centers[:,2].max()]
        ], dtype=torch.float32)
        z_min = self.scene_bbox[0, 2].item()
        z_max = self.scene_bbox[1, 2].item()
        z_len = z_max - z_min
        self.scene_bbox[0, 2] = z_min - 0.3 * z_len
        self.scene_bbox[1, 2] = z_max + 0.3 * z_len
        print("[INFO] 当前训练/渲染使用的是蓝点AABB并在z轴上下各扩展30%后的版本！")
        # 自动设置near_far
        self.near_far = [0.1, 1.5]
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # 生成与训练集分布一致的螺旋轨迹
        # 原有中心化路径（绿点）保留注释
        # print('[INFO] render_path 生成方式: get_spiral（中心化）')
        # self.render_path = torch.from_numpy(get_spiral(self.poses, self.near_far, N_views=120).astype(np.float32)).float()
        # 新方案：蓝点路径
        print('[INFO] render_path 生成方式: get_spiral_from_centers（蓝点）')
        self.render_path = torch.from_numpy(get_spiral_from_centers(camera_centers, rads_scale=1.0, N_views=120).astype(np.float32)).float()

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
            img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
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
