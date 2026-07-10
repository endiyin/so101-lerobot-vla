# SO-101 机械臂常用命令记录

> 本文档记录当前机器（Ubuntu 20.04 + LeRobot v0.4.3）的常用操作命令。  
> 当前 conda 环境名：`lerobot_py310`

---

## 1. 环境激活

```bash
conda activate lerobot_py310
```

---

## 2. 查询当前端口

### 查询机械臂串口
```bash
ls /dev/ttyACM*
```

udev 规则已固定：
- `/dev/so101_follower` → follower 从臂（序列号 `5B3D043741`）
- `/dev/so101_leader` → leader 主臂（序列号 `5B3D044803`）

### 查询相机设备
```bash
ls /dev/video*
```

当前相机：`/dev/video4`（手眼相机）

### 用官方工具查找端口
```bash
lerobot-find-port
```

---

## 3. 串口权限

### 临时权限（重启失效）
```bash
sudo chmod 666 /dev/so101_leader /dev/so101_follower
```

### 永久权限（推荐）
将当前用户加入 `dialout` 组，然后重新登录：
```bash
sudo usermod -a -G dialout $USER
newgrp dialout
```

---

## 4. 标定

### 标定从臂（follower）
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=1
```

### 标定主臂（leader）
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=1
```

### 标定文件保存位置
```bash
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/1.json
~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/1.json
```

---

## 5. 遥操作测试

### 基础遥操作（不带相机）
```bash
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=1 --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=1
```

### 带相机画面的遥操作
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower \
    --robot.id=1 \
    --robot.cameras="{'wrist': {'type': 'opencv', 'index_or_path': '/dev/video4', 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader \
    --teleop.id=1 \
    --display_data=true
```

> 遥操作前注意：
> 1. follower 从臂标定完后先断电一次再上电
> 2. 两个臂姿态保持一致，最好都处于 rest 位
> 3. 确保电源电压正确（leader 用 7.4V，follower 按版本用 7.4V 或 12V）

---

## 6. 录制数据

### 本地录制（不上传 Hugging Face）
```bash
lerobot-record \
    --robot.disable_torque_on_disconnect=true \
    --robot.type=so101_follower \
    --robot.port=/dev/so101_follower \
    --robot.id=1 \
    --robot.cameras="{'wrist': {'type': 'opencv', 'index_or_path': '/dev/video4', 'width': 640, 'height': 480, 'fps': 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/so101_leader \
    --teleop.id=1 \
    --display_data=true \
    --dataset.repo_id=hui/so101_$(date +%Y%m%d_%H%M%S) \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=20 \
    --dataset.single_task="Grab the black cube" \
    --dataset.push_to_hub=false
```

> 使用日期时间命名数据集，例如：`hui/so101_20260710_222633`

### 录制控制快捷键
- 动作完成：按 **右键** 提前结束当前轮次
- 重录当前轮次：按 **左键**
- 退出录制：按 **ESC**

### 数据集保存位置
```bash
~/.cache/huggingface/lerobot/hui/so101_$(date +%Y%m%d_%H%M%S)
```

### 重复录制时报 File exists 错误
因为使用了秒级时间戳，通常不会重名。如果确实要覆盖某个数据集：

方法一：删除旧数据集（把 `so101_20260710_222633` 替换成你要删除的名称）
```bash
rm -rf ~/.cache/huggingface/lerobot/hui/so101_20260710_222633
```

方法二：手动指定新名称
```bash
--dataset.repo_id=hui/so101_$(date +%Y%m%d_%H%M%S)_task2
```

方法三：继续追加到已有数据集
```bash
--resume=true
```

---

## 7. 训练（ACT 策略）

### 7.1 离线训练（推荐，数据集在本地）

如果训练时报 SSL/HuggingFace 连接错误，设置离线模式：

```bash
export HF_HUB_OFFLINE=1
lerobot-train \
  --dataset.repo_id=hui/so101_20260710_222633 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_20260710_222633 \
  --job_name=act_so101_20260710_222633 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### 7.2 使用 HuggingFace 国内镜像

如果需要连接 HuggingFace，使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
lerobot-train \
  --dataset.repo_id=hui/so101_20260710_222633 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_20260710_222633 \
  --job_name=act_so101_20260710_222633 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### 7.3 显存不足时减小 batch_size

```bash
export HF_HUB_OFFLINE=1
lerobot-train \
  --dataset.repo_id=hui/so101_20260710_222633 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_20260710_222633 \
  --job_name=act_so101_20260710_222633 \
  --policy.device=cuda \
  --policy.batch_size=16 \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### 7.4 训练日志

看到类似以下输出说明训练开始：
```
INFO step:100 smpl:800 ep:3 epch:0.67 loss:4.547 grdn:131.117 lr:1.0e-05
```

### 7.5 训练步数与保存频率

默认配置：
- `steps=100000`：总共训练 10 万步
- `save_freq=20000`：每 2 万步保存一个 checkpoint

对于少量数据（如 10 条），建议缩短训练步数、提高保存频率：

```bash
export HF_HUB_OFFLINE=1
lerobot-train \
  --dataset.repo_id=hui/so101_20260710_222633 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_20260710_222633 \
  --job_name=act_so101_20260710_222633 \
  --policy.device=cuda \
  --steps=10000 \
  --save_freq=2000 \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

| 参数 | 说明 | 建议 |
|---|---|---|
| `--steps` | 总训练步数 | 少量数据用 5000~10000 |
| `--save_freq` | 保存 checkpoint 的间隔 | 少量数据用 1000~2000 |

模型保存在 `outputs/train/act_so101_20260710_222633/checkpoints/` 目录下。

### 7.6 提前停止

训练过程中按 `Ctrl+C` 中断，会自动保存最后一个 checkpoint。选择 loss 最低或效果最好的 checkpoint 用于推理。

---

## 8. 推理测试

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=1 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.id=1 \
  --robot.disable_torque_on_disconnect=true \
  --robot.cameras="{'wrist': {'type': 'opencv', 'index_or_path': '/dev/video4', 'width': 640, 'height': 480, 'fps': 30}}" \
  --display_data=true \
  --dataset.single_task="Grab the black cube" \
  --policy.path=outputs/train/act_so101_20260710_222633/checkpoints/last/pretrained_model \
  --policy.device=cuda \
  --dataset.repo_id=hui/eval_so101_20260710_222633 \
  --dataset.push_to_hub=false
```

---

## 9. 常见问题

### 9.1 NumPy 导入错误
当前环境已修复。如果新环境遇到：
```bash
pip uninstall numpy -y
pip install numpy
```

### 9.2 Permission denied: '/dev/so101_follower'
串口权限不足，加入 dialout 组：
```bash
sudo usermod -a -G dialout $USER
newgrp dialout
```

### 9.3 相机分辨率设置失败
把 height 改成相机支持的分辨率，当前相机用 480：
```bash
'height': 480
```

### 9.4 主从臂端口变化
插拔后 `/dev/so101_leader` 和 `/dev/so101_follower` 可能互换。建议创建 udev 规则固定名称。

创建规则文件：
```bash
sudo nano /etc/udev/rules.d/99-robot-arms.rules
```

写入：
```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B3D043741", SYMLINK+="so101_follower"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5B3D044803", SYMLINK+="so101_leader"
```

加载规则：
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

之后使用固定名称：
```bash
/dev/so101_follower
/dev/so101_leader
```

### 9.5 舵机通信失败（There is no status packet）
1. 检查电源是否接通、电压是否正确
2. 检查 USB 线和舵机线是否插紧
3. 重新给机械臂上电
4. 检查是否有舵机闪红灯

---

## 10. 当前硬件信息

| 设备 | 串口/路径 | 序列号 |
|---|---|---|
| follower 从臂 | `/dev/so101_follower` | `5B3D043741` |
| leader 主臂 | `/dev/so101_leader` | `5B3D044803` |
| 手眼相机 | `/dev/video4` | - |
