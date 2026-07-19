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

## 2026-07-17

### 完成内容

1. **Diffusion Policy 50000 步 + image_transforms 部署验证**
   - 部署模型：`diffusion_robodojo_stack_bowls_aug/checkpoints/050000`
   - 结果：任务成功率仍约 0%，仍然频繁夹空气
   - 与无增强版本相比没有明显改善

2. **RoboDojo 评估布局机制分析**
   - 发现 `stack_bowls` 任务评估时使用 85 个预定义布局
   - 训练集（100 episode）只覆盖约 4~5 种主要放置方式
   - 评估布局的碗位、颜色/材质种类远超训练集覆盖范围
   - 结论：训练集与评估集分布严重不匹配，这是模仿学习失败的主因

3. **RoboDojo 设计意图调研**
   - 查阅 RoboDojo 论文，确认该 benchmark 目标是评估 generalist robot manipulation policies
   - 训练集仅 35 任务 × 100 trajectory = 3500 trajectory
   - Leaderboard 显示单任务 ACT 平均成功率仅 0.32%，单任务 SmolVLA 仅 0.85%
   - 更新文档说明：该 benchmark 期望多任务训练 + 预训练 VLA，而非单任务海量数据过拟合

4. **更新 README 项目目标**
   - 将项目目标从"让单任务模仿学习成功"调整为"复现并分析 VLA 模型在机器人操作 benchmark 上的表现，形成可写在简历上的完整项目"

---

## 2026-07-16

### 完成内容

1. **Diffusion Policy 20000 步训练完成**
   - 数据集：`stack_bowls` 子集（episodes 3000~3099）
   - 耗时约 52 分钟
   - loss 从 0.795 降到 0.009
   - 关闭 `image_transforms`、提高 `num_workers=8` 后 GPU 利用率从 2~3% 提升到 80%

2. **Diffusion Policy 10000 步与 20000 步部署对比**
   - 10000 步：更迷茫，经常找不到碗的位置
   - 20000 步：能完成"靠近→夹取→移动→尝试叠放"流程，但夹取不稳、提前松抓

3. **Closed-Loop 部署实验**
   - 修改 `deploy.py` 支持 closed-loop 重新规划（`replan_interval=1`）
   - 结果：流程仍正确，但成功率无明显提升
   - 唯一一次夹住碗，因抬升高度不够推开底层碗
   - 结论：问题根源不是执行流程，而是夹取物理精度本身不够

4. **Diffusion Policy 50000 步部署观察**
   - 动作流程完全正确、像模像样
   - 但夹爪夹得很浅，始终夹空气搬运空气
   - 高层策略已学会，夹爪闭合深度/物理接触细节没学好

---

## 2026-07-15

### 完成内容

1. **Diffusion Policy 50000 步无增强训练完成**
   - 配置：`num_inference_steps=5`，`steps=50000`
   - 相比 20000 步版本，训练和部署速度提升约一倍

2. **Diffusion Policy 50000 步 image_transforms 增强训练完成**
   - 启用 brightness、contrast、saturation、hue、sharpness、affine 增强
   - `num_workers=12` 缓解增强带来的 CPU 负载

3. **ACT on `stack_bowls` 子集部署**
   - 任务成功率约 0%
   - 机械臂动到一半后原地抽搐
   - 确认 ACT 无法处理多模态动作分布

---

## 2026-07-14 及之前

### 完成内容

1. **环境搭建**
   - 安装 Miniconda + `lerobot` 环境
   - 克隆 JoyandAI/lerobot v0.4.3
   - 安装 feetech 驱动和 ffmpeg 7.1.1

2. **RoboDojo 数据集准备**
   - 下载 RoboDojo v3.0 视频格式数据集（约 120GB）
   - 生成 `stack_bowls` 任务 episode 列表（3000~3099）

3. **LeRobot bug 修复**
   - 修复 `episodes` 过滤 + `EpisodeAwareSampler` 导致的 `IndexError`
   - 修改文件：`/media/endiyin/F/lerobot/src/lerobot/datasets/lerobot_dataset.py`

4. **XPolicyLab adapter 改进**
   - 修改 `model.py` 自动识别 policy 类型（act / diffusion / smolvla 等）
   - 不再硬编码 `ACTPolicy.from_pretrained`

5. **ACT 早期验证**
   - 在 demo 数据集（1 episode）上训练 ACT，验证流程
   - 在 `stack_bowls` 子集上训练 ACT 5000 步，loss 接近 0 但部署失败

---

## 备注

- 本文件随项目进展每日更新。
- 详细实验数据、训练命令、部署观察见 [`ROBODOJO_EXPERIMENTS.md`](./ROBODOJO_EXPERIMENTS.md)。
- 训练脚本合集见 `/media/endiyin/F/RoboDojo/outputs/train/train_act_demo.txt`。

---

## 附录：休息日/无进展日模板

如果某天没有跑实验或写代码，也可以更新本文件，记录状态和下一步计划。例如：

```markdown
## 2026-07-19

### 今日状态

- 休息日 / 无实验进展
- 原因：处理其他事情 / 等待 GPU / 阅读论文 / 整理思路

### 今日思考/计划

- 回顾了前几天的 Diffusion Policy 部署结果，确认核心瓶颈是训练数据分布不足
- 明天计划启动 SmolVLA 完整 100000 步训练

### 下一步

- [ ] 启动 SmolVLA 多任务完整训练
- [ ] 训练完成后部署到 stack_bowls 任务
```

保持每天记录即可，不需要每天都有代码或实验产出。
