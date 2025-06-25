import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('.')
from dataLoader.llff_json import center_poses, render_path_spiral, normalize, average_poses, get_spiral, get_spiral_from_centers

# 1. 读取 transforms.json，支持命令行参数
json_path = sys.argv[1] if len(sys.argv) > 1 else 'my_data/lego_dino/transforms.json'
with open(json_path, 'r') as f:
    meta = json.load(f)

# 2. 提取相机中心
camera_centers = []
for frame in meta['frames']:
    mat = np.array(frame['transform_matrix'])
    center = mat[:3, 3]
    camera_centers.append(center)
camera_centers = np.array(camera_centers)

# 3. 生成两种 spiral 路径
poses = np.array([np.array(f['transform_matrix']) for f in meta['frames']]).astype(np.float32)
poses_3x4 = poses[:, :3, :]
poses_centered, pose_avg_homo = center_poses(poses_3x4)
centers = poses_centered[..., 3]
scale_factor = np.max(np.linalg.norm(centers, axis=1))
poses_centered[..., 3] /= scale_factor

# 绿点AABB（中心化后的分布）
green_scene_bbox = np.array([
    np.min(centers, axis=0),
    np.max(centers, axis=0)
])
# 蓝点AABB（原始相机中心）
blue_scene_bbox = np.array([
    np.min(camera_centers, axis=0),
    np.max(camera_centers, axis=0)
])
# 拓展z轴（上下）边界各20%
blue_z_min = blue_scene_bbox[0, 2]
blue_z_max = blue_scene_bbox[1, 2]
blue_z_len = blue_z_max - blue_z_min
expanded_blue_scene_bbox = blue_scene_bbox.copy()
expanded_blue_scene_bbox[0, 2] = blue_z_min - 0.2 * blue_z_len
expanded_blue_scene_bbox[1, 2] = blue_z_max + 0.2 * blue_z_len

near_far = [0.1, 1.5]
up = normalize(poses_centered[:, :3, 1].sum(0))

# 原有螺旋路径（红色）
spiral_poses = get_spiral(poses_centered, near_far, rads_scale=1, N_views=120)
spiral_centers = spiral_poses[:, :3, 3]

# 新增：用蓝点生成的新路径（橙色）
spiral_poses_blue = get_spiral_from_centers(camera_centers, rads_scale=1.0, N_views=120)
spiral_centers_blue = spiral_poses_blue[:, :3, 3]

# 4. 可视化aabb和near/far

def draw_aabb(ax, aabb_min, aabb_max, color='k', label=None):
    for s, e in [
        ([0,0,0],[1,0,0]), ([0,0,0],[0,1,0]), ([1,0,0],[1,1,0]), ([0,1,0],[1,1,0]),
        ([0,0,1],[1,0,1]), ([0,0,1],[0,1,1]), ([1,0,1],[1,1,1]), ([0,1,1],[1,1,1]),
        ([0,0,0],[0,0,1]), ([1,0,0],[1,0,1]), ([0,1,0],[0,1,1]), ([1,1,0],[1,1,1])
    ]:
        ax.plot(
            [aabb_min[0]+(aabb_max[0]-aabb_min[0])*s[0], aabb_min[0]+(aabb_max[0]-aabb_min[0])*e[0]],
            [aabb_min[1]+(aabb_max[1]-aabb_min[1])*s[1], aabb_min[1]+(aabb_max[1]-aabb_min[1])*e[1]],
            [aabb_min[2]+(aabb_max[2]-aabb_min[2])*s[2], aabb_min[2]+(aabb_max[2]-aabb_min[2])*e[2]],
            color=color, linewidth=1.5, label=label if s==[0,0,0] else None
        )

def draw_sphere(ax, center, radius, color, alpha=0.1):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha)

# 5. 绘图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2], c='b', label='Original Camera Centers')
centered_centers = np.stack([poses_centered[:, 0, 3], poses_centered[:, 1, 3], poses_centered[:, 2, 3]], axis=1)
ax.scatter(centered_centers[:,0], centered_centers[:,1], centered_centers[:,2], c='g', label='Centered Camera Centers')
ax.plot(spiral_centers[:,0], spiral_centers[:,1], spiral_centers[:,2], c='r', label='Spiral Render Path (centered)')
ax.plot(spiral_centers_blue[:,0], spiral_centers_blue[:,1], spiral_centers_blue[:,2], c='orange', label='Spiral Path (blue points)')
draw_aabb(ax, green_scene_bbox[0], green_scene_bbox[1], color='k', label='AABB (centered)')
draw_aabb(ax, blue_scene_bbox[0], blue_scene_bbox[1], color='c', label='AABB (blue points)')
draw_aabb(ax, expanded_blue_scene_bbox[0], expanded_blue_scene_bbox[1], color='m', label='AABB (blue+z20%)')
draw_sphere(ax, np.mean(green_scene_bbox, axis=0), near_far[0], color='c', alpha=0.2)
draw_sphere(ax, np.mean(green_scene_bbox, axis=0), near_far[1], color='m', alpha=0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Camera Centers, Spiral Path, Green/Blue AABB, Near/Far')
plt.show()