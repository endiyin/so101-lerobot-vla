# 每日进度记录

> 本文件按日期记录项目进展，保持每天更新。

---

## 2026-07-18（今天）

### 完成内容

1. **SmolVLA 依赖安装与模型加载验证**
   - 安装 `lerobot[smolvla]` 依赖
   - 验证 `lerobot/smolvla_base` 可正常加载：总参数 450M，可训练参数 100M

2. **修复 SmolVLA `rename_map` 导致 `image_features` 不匹配问题**
   - 问题：`make_policy()` 用数据集特征填充 `cfg.input_features`（key 为 `cam_high/...`），但运行时 preprocessor 把 batch key 改为 `camera1/2/3`，导致 `prepare_images()` 找不到图像
   - 修复：修改 `/media/endiyin/F/lerobot/src/lerobot/policies/factory.py`，在 `make_policy()` 中对 `cfg.input_features` 应用 `rename_map`
   - 修复前训练启动即报 `ValueError: All image features are missing from the batch`
   - 修复后训练正常启动

3. **SmolVLA 1000 步短测试通过**
   - 数据集：RoboDojo 全量 35 任务 / 3500 episode
   - 配置：`batch_size=2`，`num_workers=4`，冻结视觉编码器，只训练 action expert
   - 结果：
     - 耗时约 2 分 51 秒
     - loss 从 0.568 降到 0.204
     - 显存占用约 2.8 GB / 24 GB
     - 无 OOM
   - 结论：SmolVLA 在 RoboDojo 多任务数据上训练流程正常，`batch_size=2` 在 RTX 3090 上显存安全

4. **文档更新**
   - `ROBODOJO_EXPERIMENTS.md` 新增 3.3 节（rename_map 修复）和 4.6 节（SmolVLA 短测试结果）
   - `README.md` 更新实验状态表、对比表、近期目标
   - `train_act_demo.txt` 新增 4.1 节短测试命令
   - 新建 `outputs/train/smolvla_robodojo_test/README.md` 训练记录卡
   - 创建本 `DAILY_PROGRESS.md` 每日进度文件

5. **Git 提交**
   - `so101-lerobot-vla`：已 push 文档更新到 GitHub
   - `lerobot`：`factory.py` 和 `lerobot_dataset.py` 已本地 commit，未 push 到共享仓库

### 下一步计划

- 启动 SmolVLA 完整 100000 步多任务训练
- 训练完成后部署到 `stack_bowls` 任务，与 Diffusion Policy 单任务结果对比

---

## 2026-07-20（今天）

### 完成内容

1. **启动 SmolVLA 完整多任务训练**
   - 脚本：`/media/endiyin/F/RoboDojo/outputs/train/train_smolvla_multitask.sh`
   - 日志：`/media/endiyin/F/RoboDojo/outputs/train/smolvla_robodojo_multitask.log`
   - 数据集：RoboDojo 全量 35 任务 / 3500 episode / 1,859,602 帧
   - 配置：`batch_size=2`，`steps=100000`，`save_freq=10000`，冻结视觉编码器，只训练 action expert
   - 模型输出：`/media/endiyin/F/RoboDojo/outputs/train/smolvla_robodojo_multitask`
   - 启动时间：2026-07-20 13:54:26
   - 预计总耗时：约 4~5 小时

2. **监控训练状态**
   - 训练正常启动，GPU 利用率约 90%
   - 显存占用约 2.7 GB / 24 GB
   - 无 OOM

### 下一步计划

- 等待 100000 步训练完成
- 每 10000 步会保存一个 checkpoint
- 训练完成后部署到 `stack_bowls` 任务评估

---

## 2026-07-19

### 今日状态

- 休息日 / 无实验进展

### 下一步

- 启动 SmolVLA 完整多任务训练

---

## 2026-07-17

### 完成内容

1. **Diffusion Policy 50000 步 + image_transforms 部署验证**
   - 部署模型：`/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_aug/checkpoints/050000/pretrained_model`
   - 配置：`num_inference_steps=5`，训练时启用 brightness/contrast/saturation/hue/sharpness/affine 增强
   - 结果：任务成功率仍约 0%，仍然频繁夹空气
   - 与无增强版本相比没有明显改善
   - 结论：image_transforms 无法弥补训练集与评估集分布不匹配的问题

2. **RoboDojo 评估布局机制分析**
   - 评估布局目录：`/media/endiyin/F/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/`
   - 发现 `stack_bowls` 任务评估时使用 85 个预定义布局（`stack_bowls_0.json` ~ `stack_bowls_84.json`）
   - 每个 episode 按顺序选取一个布局文件
   - 布局中碗的 `default_pos` 固定，`rotate_rand=false`
   - 训练集（100 episode，3000~3099）只覆盖约 4~5 种主要放置方式
   - 评估布局中碗的 `category_idx` 分布很广（1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15），对应不同颜色/材质
   - 结论：训练集与评估集分布严重不匹配，这是模仿学习失败的主因

3. **RoboDojo 设计意图调研**
   - 查阅 RoboDojo 论文 [RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies](https://arxiv.org/html/2607.04434v3)
   - 确认该 benchmark 目标是评估 generalist robot manipulation policies，不是单任务模仿学习
   - 训练集仅 35 任务 × 100 trajectory = 3500 trajectory
   - Leaderboard 数据：
     - Hy-Embodied-0.5-VLA：Generalization 8.39%，平均 8.80%
     - π0.5：Generalization 8.17%，平均 6.91%
     - SmolVLA (Single Task)：Generalization 1.22%，平均 0.85%
     - ACT (Single-Task)：Generalization 0.56%，平均 0.32%
   - 更新文档说明：该 benchmark 期望多任务训练 + 预训练 VLA，而非单任务海量数据过拟合

4. **更新 README 项目目标**
   - 将项目目标从"让单任务模仿学习成功"调整为"复现并分析 VLA 模型在机器人操作 benchmark 上的表现，形成可写在简历上的完整项目"
   - 把后续重点转向 SmolVLA / π0 等 VLA 模型

---

## 2026-07-16

### 完成内容

1. **Diffusion Policy 20000 步训练完成**
   - 数据集：`stack_bowls` 子集（episodes 3000~3099），共 44874 帧
   - 训练命令见 `train_act_demo.txt` 第 2 节
   - 开始时间：2026-07-16 16:33:54
   - 结束时间：2026-07-16 17:26:14
   - 总耗时：约 52 分钟
   - loss 从 0.795 降到 0.009
   - 模型参数量：271M
   - 关键优化：关闭 `image_transforms`、设置 `num_workers=8` 后，GPU 利用率从 2~3% 提升到 80%，解决视频解码 CPU 瓶颈
   - Checkpoint：`/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls/checkpoints/020000/pretrained_model`

2. **Diffusion Policy 10000 步与 20000 步部署对比**
   - 10000 步（`checkpoints/010000`）：更迷茫，经常找不到碗的位置
   - 20000 步（`checkpoints/020000`）：能完成"靠近→夹取→移动→尝试叠放"流程，但夹取不稳、提前松抓
   - 动作速度较慢，因为 `num_inference_steps=10`

3. **Closed-Loop 部署实验**
   - 修改 `deploy.py` 支持 closed-loop 重新规划
   - 配置：`replan_interval=1`（每执行 1 个 action 重新调用 `get_action`）
   - 使用模型：`diffusion_robodojo_stack_bowls_fast/checkpoints/050000`
   - 结果：流程仍正确，但成功率无明显提升
   - 唯一一次成功夹住碗并移动，但因抬升高度不够推开底层碗
   - 结论：问题根源不是执行流程，而是夹取物理精度本身不够

4. **Diffusion Policy 50000 步部署观察**
   - 使用模型：`/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_fast/checkpoints/050000/pretrained_model`
   - 配置：`num_inference_steps=5`
   - 动作流程完全正确、像模像样
   - 但夹爪夹得很浅，始终夹空气搬运空气
   - 高层策略已学会，夹爪闭合深度/物理接触细节没学好
   - 如果夹爪把碗碰到未见过的位置，夹臂会一直抽搐直到环境重启
   - 即使没夹住东西，也会继续移动到叠放位置，说明只是模仿动作流程

---

## 2026-07-15

### 完成内容

1. **Diffusion Policy 50000 步无增强训练完成**
   - 模型：`diffusion_robodojo_stack_bowls_fast`
   - 配置：`num_inference_steps=5`，`steps=50000`，`batch_size=16`，`num_workers=8`
   - 相比 20000 步版本（`num_inference_steps=10`），训练和部署速度提升约一倍
   - Checkpoint：`/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_fast/checkpoints/050000/pretrained_model`
   - 脚本：`/media/endiyin/F/RoboDojo/outputs/train/train_diffusion_stack_bowls_fast.sh`

2. **Diffusion Policy 50000 步 image_transforms 增强训练完成**
   - 模型：`diffusion_robodojo_stack_bowls_aug`
   - 启用增强：brightness、contrast、saturation、hue、sharpness、affine
   - 配置：`num_workers=12` 缓解增强带来的 CPU 负载
   - Checkpoint：`/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_aug/checkpoints/050000/pretrained_model`
   - 创建模型训练记录卡 `outputs/train/diffusion_robodojo_stack_bowls_aug/README.md`

3. **ACT on `stack_bowls` 子集部署**
   - 模型：`act_robodojo_stack_bowls`（训练 5000 步，loss 接近 0）
   - 任务成功率约 0%
   - 机械臂动到一半后原地抽搐
   - 确认 ACT 无法处理多模态动作分布：MSE 损失把多个正确策略平均成"四不像"

---

## 2026-07-14 及之前

### 完成内容

1. **环境搭建**
   - 安装 Miniconda（Python 3.10）
   - 创建 `lerobot` conda 环境
   - 克隆 JoyandAI/lerobot v0.4.3 到 `/media/endiyin/F/lerobot`
   - 安装 lerobot 依赖：`pip install -e ".[feetech]"`
   - 安装 ffmpeg 7.1.1：`conda install -c conda-forge ffmpeg=7.1.1`
   - 配置 pip 清华镜像源加速下载

2. **RoboDojo 数据集准备**
   - 下载 RoboDojo v3.0 视频格式数据集到 `/media/endiyin/F/RoboDojo/data/RoboDojo_lerobot_v30_video`（约 120GB）
   - 数据集包含 35 个任务、3500 个 episode、约 186 万帧
   - 生成 `stack_bowls` 任务 episode 列表：`/media/endiyin/F/RoboDojo/outputs/train/stack_bowls_episodes.txt`（episodes 3000~3099）

3. **LeRobot bug 修复**
   - 问题：使用 `--dataset.episodes` 过滤子集训练 Diffusion Policy 时，启动后立刻崩溃：
     ```text
     IndexError: Invalid key: 1634679 is out of bounds for size 44874
     ```
   - 根因：`EpisodeAwareSampler` 返回整个数据集的绝对帧索引，但过滤后的 `hf_dataset` 行数更少
   - 修复：修改 `/media/endiyin/F/lerobot/src/lerobot/datasets/lerobot_dataset.py`，在 `__getitem__` 中将绝对索引映射到过滤数据集的相对索引

4. **XPolicyLab adapter 改进**
   - 修改文件：`/media/endiyin/F/RoboDojo/XPolicyLab/policy/robodojo_act_lerobot/model.py`
   - 原代码硬编码 `ACTPolicy.from_pretrained`，无法加载 Diffusion Policy checkpoint
   - 改进为从 checkpoint 读取 `PreTrainedConfig`，根据 `config.type` 动态获取 policy 类（act / diffusion / smolvla 等）
   - 现在同一个 adapter 可部署多种 policy

5. **ACT 早期验证**
   - Demo 数据集：`/media/endiyin/F/RoboDojo/data/demo_arx_x5_white_bow`（1 episode，241 帧）
   - 在 demo 上训练 ACT 5000 步验证流程，最终 loss 约 0.07
   - 部署时机械臂抽搐，原因是数据量极小 + ACT 对多模态动作平均
   - 在 `stack_bowls` 子集（100 episode）上训练 ACT 5000 步，loss 接近 0 但部署成功率约 0%
   - 确认 ACT 不适合该任务的多模态动作分布

---

## 备注

- 本文件随项目进展每日更新。即使某天没做实验，也可以写一句「今日休息/无进展 + 下一步计划」，保持记录连续性。
- 详细实验数据、训练命令、部署观察见 [`ROBODOJO_EXPERIMENTS.md`](./ROBODOJO_EXPERIMENTS.md)。
- 训练脚本合集见 `/media/endiyin/F/RoboDojo/outputs/train/train_act_demo.txt`。
