# SO-101 + LeRobot VLA 实践

> 基于 [LeRobot](https://github.com/huggingface/lerobot) 框架和 SO-101 低成本机械臂，复现 VLA（Vision-Language-Action）相关算法，用于具身智能方向实习准备。

---

## 项目简介

本项目记录我在 Ubuntu 20.04 + SO-101 双臂机械臂上，使用 LeRobot v0.4.3 完成数据采集、训练、部署的全流程实践。

目前主要聚焦：
- 机械臂标定与遥操作
- 模仿学习 / 行为克隆
- VLA 模型复现与对比

后续会持续补充 SmolVLA、π0、π0-FAST 等 VLA 模型的实验结果。

---

## 硬件配置

| 组件 | 型号/参数 |
|---|---|
| 操作系统 | Ubuntu 20.04 |
| GPU | NVIDIA RTX 4060 Laptop |
| 主臂（Leader） | SO-101 Leader，7.4V 飞特舵机 |
| 从臂（Follower） | SO-101 Follower，7.4V/12V 飞特舵机 |
| 舵机 | 飞特 STS3215 系列 |
| 手眼相机 | OpenCV USB 广角摄像头 |
| 相机分辨率 | 640 × 480 @ 30fps |
| 固定串口 | `/dev/so101_leader`, `/dev/so101_follower` |
| 相机设备 | `/dev/video4` |

---

## 已完成的实验

| 算法 | 状态 | 数据集 | 模型路径 | 备注 |
|---|---|---|---|---|
| ACT | ✅ 已完成 | `hui/so101_20260710_222633` | `outputs/train/act_so101_20260710_222633` | loss 6.67 → 0.15 |
| Diffusion Policy | ✅ 已完成 | RoboDojo `stack_bowls` 子集 | `outputs/train/diffusion_robodojo_stack_bowls` | 详见 [ROBODOJO_EXPERIMENTS.md](./ROBODOJO_EXPERIMENTS.md) |
| Diffusion Policy（高效配置） | ✅ 已完成 | RoboDojo `stack_bowls` 子集 | `outputs/train/diffusion_robodojo_stack_bowls_fast` | 50000 步，num_inference_steps=5 |
| Diffusion Policy（image_transforms 增强） | ✅ 已完成训练 | RoboDojo `stack_bowls` 子集 | `outputs/train/diffusion_robodojo_stack_bowls_aug` | 50000 步，开启数据增强，部署成功率仍约 0%，根因是训练集与评估集分布不匹配 |
| SmolVLA | ✅ 短测试通过 | RoboDojo 全量 35 任务 | `outputs/train/smolvla_robodojo_test` | 1000 步多任务短测试，loss 0.568 → 0.204，显存 2.8GB，待启动完整 100000 步训练 |
| π0 | ⏳ 待进行 | | | |
| π0-FAST | ⏳ 待进行 | | | |

---

## 训练结果对比

| 算法 | 数据量 | 训练步数 | 最终 loss | 任务成功率 | 行为表现 | 单步延迟 | 显存占用 | 备注 |
|---|---|---|---|---|---|---|---|---|
| ACT | 10 eps | 10K | 0.15 | **原位置 90% / 偏移 5cm 5%** | 真机抓取方块较稳 | - | - | 过拟合，位置敏感 |
| ACT (RoboDojo) | 100 eps | 5K | ~0.000 | ~0% | 动到一半原地抽搐，不知如何继续 | - | - | 多模态动作平均导致策略混乱 |
| Diffusion 10K (RoboDojo) | 100 eps | 10K | - | ~0% | 更迷茫，经常找不到碗的位置 | 慢 | 7GB / 24GB | 训练不足 |
| Diffusion 20K (RoboDojo) | 100 eps | 20K | 0.009 | ~0% | 能完成夹取→移动→尝试叠放流程，但夹取不稳、提前松抓 | 慢 | 7GB / 24GB | 学到任务结构，精度不足 |
| Diffusion Fast 50K (RoboDojo) | 100 eps | 50K | - | ~0% | 随机位置下流程完全正确、尽力夹取，但夹爪夹得很浅，始终夹空气搬运空气 | 快 | 7GB / 24GB | 高层策略已学会，夹爪闭合精度不足 |
| Diffusion Fast 50K + Closed-Loop (RoboDojo) | 100 eps | 50K | - | ~0% | 流程仍正确，但成功率无明显提升；唯一一次夹住因抬升不够高推开底层碗 | 慢 | 7GB / 24GB | Closed-loop 只修正执行流程，未提升夹取物理精度 |
| Diffusion Fast 50K + image_transforms (RoboDojo) | 100 eps | 50K | 待补充 | ~0% | 仍然夹空气，无法精准定位碗的位置 | 快 | 7GB / 24GB | 训练集只覆盖 4~5 种布局，评估有 85 种预定义布局，分布不匹配 |
| SmolVLA 1000-step test (RoboDojo) | 3500 eps (35 tasks) | 1K | 0.204 | 未部署 | 训练启动正常，loss 下降，无 OOM | - | 2.8GB / 24GB | 多任务短测试通过；完整训练与部署待进行 |

### 当前核心困难

**不是算法问题，是数据分布问题。**

RoboDojo `stack_bowls` 的评估集有 **85 个预定义布局**（即 85 种不同的碗放置位置配置），但当前训练集只用了 100 个 episode，大约只覆盖了 4~5 种主要放置方式。对模仿学习来说，这是极其困难的设定：

- **ACT**：用 MSE 损失平均多模态动作，数据少时容易抽搐、迷茫
- **Diffusion Policy**：虽然能表达多模态分布，但仍然依赖训练数据覆盖评估分布
- 271M 参数的模型用 100 个 episode 训练，严重欠拟合位置分布

类比理解：只给你看 4~5 张不同桌子的照片学做菜，考试时给你 85 张完全不同的桌子，还要准确拿到指定杯子——人也会懵。

**所以目前所有模型（ACT / Diffusion 20K / Diffusion 50K / Diffusion 50K + image_transforms）成功率都接近 0% 的根因是训练数据没有覆盖评估分布，而不是训练步数、开环/闭环或数据增强不够。**

### 为什么 RoboDojo 要这样设计？

这不是 RoboDojo 的 bug，而是它的**设计意图**。

RoboDojo 论文 [RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/html/2607.04434v3) 明确指出：

- RoboDojo 是评估 **generalist robot manipulation policies** 的 benchmark，不是单任务模仿学习
- 训练集设计为 **35 个任务 × 100 个 trajectory = 3500 个 trajectory**，覆盖 Generalization、Memory、Long-Horizon、Precision 四个维度
- 评估时的 Generalization 维度专门测试对**未见过的背景、光照、杂物、目标物体**的鲁棒性

官方 leaderboard 也证实了单任务模仿学习在这个 benchmark 上的表现：

| 模型 | Generalization 成功率 | 平均成功率 |
|---|---|---|
| Hy-Embodied-0.5-VLA（最好的 VLA） | 8.39% | 8.80% |
| π0.5 | 8.17% | 6.91% |
| SmolVLA (Single Task) | 1.22% | 0.85% |
| ACT (Single-Task) | 0.56% | 0.32% |

也就是说，**单任务 ACT 在 RoboDojo 上平均成功率本来就只有 0.32%**。我们的实验结果完全符合这个 benchmark 对单任务模仿学习的预期。

**RoboDojo 期望的参赛方式**是多任务训练 + 预训练 VLA + 数据增强 + domain randomization，而不是单任务训练一个 Diffusion Policy。

### ACT 实验分析

- **数据集**：10 条 `Grab the black cube` 任务数据
- **数据采集方式**：在 10 次采集中，已尽量让目标方块在初始位置上下左右偏移约 5cm 进行采集，以增加位置多样性
- **现象**：在训练分布内的位置，推理成功率可达 **90%**
- **问题**：当方块移动到训练分布外的位置（或从未出现过的组合姿态）时，成功率骤降至 **5%** 左右
- **原因**：尽管考虑了位置变化，但 10 条数据量仍然过小，ACT 模型无法充分学习视觉-动作映射的泛化能力，存在明显过拟合
- **改进方向**：
  - 扩充数据集到 50-100 条，覆盖更多位置、姿态、遮挡和光照条件
  - 引入数据增强（颜色抖动、仿射变换等）
  - 增加 front 环境相机，提供全局视角
  - 尝试 Diffusion / VLA 等更具泛化性的策略

---

## 快速开始

### 1. 环境准备

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot
git clone https://github.com/JoyandAI/lerobot.git
cd lerobot
pip install -e ".[feetech]"
conda install -c conda-forge ffmpeg=7.1.1 -y
```

### 2. 标定机械臂

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/so101_follower --robot.id=1
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/so101_leader --teleop.id=1
```

### 3. 数据采集

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

### 4. 训练 ACT

```bash
export HF_HUB_OFFLINE=1
lerobot-train \
  --dataset.repo_id=hui/so101_20260710_222633 \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_20260710_222633 \
  --job_name=act_so101_20260710_222633 \
  --policy.device=cuda \
  --steps=10000 \
  --save_freq=500 \
  --policy.push_to_hub=false \
  --wandb.enable=false
```

### 5. 推理部署

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
  --dataset.repo_id=hui/eval_so101_$(date +%Y%m%d_%H%M%S) \
  --dataset.push_to_hub=false
```

完整命令参考 [`SO101_COMMANDS.md`](./SO101_COMMANDS.md)。

---

## 踩坑记录

详见 [`SO101_COMMANDS.md`](./SO101_COMMANDS.md) 中的 **常见问题** 章节，主要包括：

- GraalPy 环境下 NumPy 编译失败 → 重建标准 CPython 环境
- Linux 串口权限不足 → `dialout` 用户组
- USB 设备号漂移 → udev 规则固定串口名
- HuggingFace SSL 错误 → `HF_HUB_OFFLINE=1`
- 相机分辨率设置失败 → 改用相机支持的分辨率
- rerun viewer 图像只显示一帧 → 修改 `visualization_utils.py` 的 `static=True` 为 `static=False`
- 6 号舵机过载 → 检查电源、避免长时间夹持硬物

---

## 后续计划

### 近期目标（面向 VLA 实习准备）

当前项目核心目标已从"让单任务模仿学习成功"转向"复现并分析 VLA 模型在机器人操作 benchmark 上的表现，形成可写在简历上的完整项目"。

1. **复现 SmolVLA 在 RoboDojo 上的微调**
   - ✅ 从 HuggingFace 加载 `lerobot/smolvla_base`（或 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`）
   - ✅ 在 RoboDojo 多任务数据上微调 action expert（1000 步短测试通过）
   - ✅ 解决 HuggingFace 下载、显存、数据格式匹配等问题
   - ⏳ 部署到 `stack_bowls` 任务并记录表现（需先完成完整训练）

2. **复现 π0 / π0.5（可选，资源允许）**
   - 在 RTX 3090 24GB 上测试可行性
   - 如显存不足，尝试 LoRA / gradient checkpointing / bfloat16

3. **多模型对比与消融分析**
   - ACT（单任务） vs Diffusion Policy（单任务/多任务） vs SmolVLA
   - 对比指标：任务成功率、推理延迟、显存占用、训练耗时
   - 重点分析：VLA 是否比纯模仿学习在泛化任务上更有优势

4. **整理技术博客 / 项目文档**
   - 把实验过程、踩坑记录、对比结果整理成博客或 README
   - 突出工程贡献：LeRobot bug 修复、XPolicyLab adapter 改进

### SO-101 真机方向（长期）

- [ ] 扩充真机数据集到 50-100 条
- [ ] 增加 front 环境相机
- [ ] 在 SO-101 上部署 VLA 模型
- [ ] 真机多任务能力验证

### 已完成

- [x] 在 SO-101 真机上完成 ACT 数据采集与训练
- [x] 在 RoboDojo 仿真上完成 ACT / Diffusion Policy 单任务训练与部署
- [x] 分析 RoboDojo 评估布局机制与单任务模仿学习失败根因
- [x] 修复 LeRobot episodes 过滤导致的 IndexError
- [x] 改进 XPolicyLab adapter 支持自动识别 policy 类型
- [x] 完成 SmolVLA 在 RoboDojo 多任务数据上的 1000 步短测试
- [x] 修复 SmolVLA `rename_map` 导致 `image_features` 不匹配的 bug

---

## 参考

- [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
- [JoyandAI/LeRobot 适配仓库](https://github.com/JoyandAI/lerobot)
- [SO-101 官方文档](https://github.com/TheRobotStudio/SO-ARM100)
- [ACT 论文](https://arxiv.org/abs/2304.13705)
- [π0 论文](https://arxiv.org/abs/2410.24164)
- [SmolVLA](https://huggingface.co/blog/smolvla)

---

> 本项目用于 VLA / 具身智能方向实习准备，欢迎交流指正。
