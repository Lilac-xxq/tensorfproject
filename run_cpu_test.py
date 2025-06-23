import os
import shlex
import subprocess

# ==============================================================================
#                      CPU 快速测试启动脚本 (Smoke Test)
# ==============================================================================
#
#   运行方式:
#   在 TensoRF/ 根目录下，直接执行 `python run_cpu_test.py`
#
#   功能:
#   1. 强制使用 CPU 运行。
#   2. 自动创建一份超轻量级配置文件，用于快速验证。
#   3. 快速完成一个极短的训练流程。
#   4. 快速渲染一个（效果会很差的）环绕视频。
#   5. 将所有产物保存在 `log/cpu_smoke_test` 目录下，不污染其他实验。
#
# ==============================================================================

# --- 步骤 0: 强制 PyTorch 使用 CPU ---
# 这是确保在 CPU 上运行的关键
print("INFO: Setting environment to force CPU usage.")
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# --- 步骤 1: 创建一个超轻量级的临时配置文件 ---
config_filename = 'configs/cpu_smoke_test.txt'
exp_name = 'cpu_smoke_test'

# 这是一个极度缩减的配置，专为快速跑通流程设计
config_content = f"""
dataset_name = llff_json
datadir = ./my_data/lego_dino
expname = {exp_name}
basedir = ./log

# 为模型组件数量设置默认值 (修复 'NoneType' 和 'IndexError' 的关键)
n_lamb_sigma = [8, 8, 8]
n_lamb_sh = [24, 24, 24]

# 大幅减少迭代次数
n_iters = 100
# 减小批大小
batch_size = 1024
# 增加下采样率，让图片变得更小，光线总数减少
downsample_train = 8

# 大幅降低模型分辨率
N_voxel_init = 32768
N_voxel_final = 32768
# 关键修复：将提升列表设置为一个在100次迭代中永远达不到的数字。
# 这样既能避免 train.py 因接收到 None 而崩溃，又能保证在测试中不触发该功能。
upsamp_list = [999]
update_AlphaMask_list = [999]

# 强制使用经验证的、无BUG的着色模式，以绕开默认模式下的隐藏BUG
shadingMode = MLP_Fea

# 关闭训练过程中的可视化，以节约时间
N_vis = -1      # 评测所有test


# 训练后仍然渲染测试集和路径视频，以验证完整流程
render_test = 1
"""

print(f"INFO: Creating temporary config file: {config_filename}")
with open(config_filename, 'w') as f:
    f.write(config_content)


def run_command(command):
    """一个辅助函数，用于执行命令行指令并打印输出。"""
    print(f"🚀 Executing: {command}")
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout is not None:
        for line in process.stdout:
            print(line.strip())
    process.wait()
    if process.returncode != 0:
        print(f"❌ Command failed with return code {process.returncode}")
    else:
        print("✅ Command finished successfully.")
    return process.returncode


# --- 步骤 2: 执行快速训练 ---
print("\n" + "="*50)
print("🏋️‍♀️ Starting smoke test training on CPU... (This should be fast)")
print("="*50)
training_command = f"python train.py --config {config_filename}"
run_command(training_command)


# --- 步骤 3: 执行快速渲染 ---
print("\n" + "="*50)
print("🎥 Starting smoke test rendering on CPU...")
print("="*50)

checkpoint_path = f'./log/{exp_name}/{exp_name}.th'

if os.path.exists(checkpoint_path):
    rendering_command = f"python train.py --config {config_filename} --ckpt {checkpoint_path} --render_only 1 --render_path 1"
    run_command(rendering_command)
else:
    print(f"❌ ERROR: Could not find checkpoint at {checkpoint_path}. Skipping rendering.")

print("\n" + "="*50)
print("🎉 CPU smoke test finished! 🎉")
print(f"Check the outputs (model, logs, and a low-quality video) in the directory: ./log/{exp_name}/")
print("="*50)

# 可选：删除临时配置文件
# os.remove(config_filename)
# print(f"INFO: Removed temporary config file: {config_filename}") 