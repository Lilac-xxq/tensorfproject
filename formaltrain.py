!pip install configargparse
!pip install plyfile
!pip install lpips
!pip install imageio
!pip install imageio-ffmpeg

import shutil
import os

# 删除整个logs目录及其所有内容
shutil.rmtree('/kaggle/working', ignore_errors=True)

# 如果只想删除某个子目录，比如 formal_train_tensorf
# shutil.rmtree('/kaggle/working/logs/formal_train_tensorf', ignore_errors=True)

# 如果只想删除 summaries
# shutil.rmtree('/kaggle/working/logs/summaries', ignore_errors=True)

import os
import shlex
import subprocess

# ==============================================================================
#                      Kaggle Notebook 启动器
# ==============================================================================

# --- 关键步骤：切换工作目录 ---
# Kaggle Notebook 的默认工作目录是 /kaggle/working/
# 我们需要切换到 TensoRF 项目的根目录，这样脚本才能找到 train.py 和数据。
# 请确保下面的路径和您在右侧看到的 'Input' 文件路径完全一致。
project_path = '/kaggle/input/tensorffinaltrainv5/TensoRF'

print(f"原始工作目录: {os.getcwd()}")
try:
    os.chdir(project_path)
    print(f"成功切换到工作目录: {os.getcwd()}")
except FileNotFoundError:
    print(f"❌ 错误：找不到项目路径 '{project_path}'。请检查路径是否正确。")
    # 如果路径错误，则不继续执行
    exit()


# --- 1. 高质量训练参数定义 ---
# 所有参数都在这里定义，不依赖外部 config 文件。
args = {
    # 数据集配置
    "dataset_name": "llff_json",
    "datadir": "./my_data/lego_dino",
    "expname": "dino_final_kaggle",
    "basedir": "/kaggle/working/log",

    # 训练设置
    "n_iters": 30000,
    "batch_size": 2048,  # ★★★ 修改点：从 4096 降到 2048，减少显存压力 ★★★
    "downsample_train": 1.0,
    "downsample_test": 1.0,

    # 学习率
    "lr_init": 0.02,
    "lr_basis": 0.001,
    "lr_decay_iters": -1,
    "lr_decay_target_ratio": 0.1,
    "lr_upsample_reset": 1,

    # 正则化权重
    "L1_weight_inital": 1e-4,
    "L1_weight_rest": 5e-5,
    "Ortho_weight": 0.0,
    "TV_weight_density": 0.0,
    "TV_weight_app": 0.0,

    # 为模型组件数量设置默认值
    "n_lamb_sigma": [16, 16, 16],
    "n_lamb_sh": [48, 48, 48],

    # 模型分辨率
    "N_voxel_init": 2097152,  # 128**3
    "N_voxel_final": 27000000, # 300**3

    # 上采样计划
    "upsamp_list": [2000, 3000, 4000, 5500, 7000],
    "update_AlphaMask_list": [4000,8000],

    # 渲染器与特征设置
    "shadingMode": "MLP_Fea",
    "pos_pe": 6,
    "view_pe": 6,
    "fea_pe": 6,
    "featureC": 128,
    "data_dim_color": 27,
    
    # 评估与可视化
    "N_vis": -1,
    "vis_every": 10000,
    "render_test": 1,
}

# --- 2. 辅助函数，用于执行和打印命令 ---
def run_command(command):
    """一个辅助函数，用于执行命令行指令并实时打印输出。"""
    print(f"🚀 Executing: {command}\n")
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ''):
        print(line.strip())
    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Command failed with return code {process.returncode}")
    else:
        print("\n✅ Command finished successfully.")
    return process.returncode

# --- 3. 构建并执行训练命令 ---
print("\n" + "="*50)
print("🚀 (1/2) 开始正式训练... (这可能需要数小时)")
print("="*50)

# ==============================================================================
#                      ★★★  这里是唯一的修改点 ★★★
# ==============================================================================
def arg_to_str(key, value):
    """将Python的list转成action='append'参数能接受的命令行格式"""
    if isinstance(value, list):
        # 对于 action="append" 的参数，每个值都要单独加上 --key
        return " ".join([f"--{key} {v}" for v in value])
    else:
        return f"--{key} {value}"

train_args_str = " ".join([arg_to_str(key, value) for key, value in args.items()])
train_command = f"python train.py {train_args_str}"

if run_command(train_command) != 0:
    print("❌ 训练失败，后续步骤已取消。")
    exit()

print("✅ (1/2) 训练完成！")


# --- 4. 构建并执行渲染命令 ---
print("\n" + "="*50)
print("🎥 (2/2) 开始渲染最终的平滑环绕视频...")
print("="*50)

# 定位训练好的模型文件
checkpoint_path = os.path.join(args['basedir'], args['expname'], f"{args['expname']}.th")

if not os.path.exists(checkpoint_path):
    print(f"❌ 错误：找不到训练好的模型文件: {checkpoint_path}")
    print("渲染步骤已跳过。")
    exit()

# 在原有参数基础上，增加渲染特定参数
render_args = {
    "ckpt": checkpoint_path,
    "render_only": 1,
    "render_path": 1,
}
# render_args_str = " ".join([f"--{key} {value}" for key, value in render_args.items()])  # 这行不再需要

final_args = args.copy()
final_args.update(render_args)
# 直接使用上面定义好的 arg_to_str 函数
final_args_str = " ".join([arg_to_str(key, value) for key, value in final_args.items()])
render_command = f"python train.py {final_args_str}"


if run_command(render_command) != 0:
    print("❌ 渲染失败。")
    exit()

print("✅ (2/2) 渲染完成！")
print("\n🎉 全部流程成功结束！")
# Kaggle的输出文件会保存在 /kaggle/working/ 目录下，所以我们打印完整的输出路径
output_dir = os.path.abspath(os.path.join(args['basedir'], args['expname']))
print(f"所有输出（模型、日志、视频）均位于目录: {output_dir}")
print("==================================================")