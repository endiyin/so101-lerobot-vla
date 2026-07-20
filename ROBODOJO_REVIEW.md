# RoboDojo 仿真实验复盘：从模仿学习到 VLA 的失败与收获

> 本文是我在 RoboDojo 仿真 benchmark 上复现 ACT、Diffusion Policy、SmolVLA 三种策略后的完整复盘。  
> 适合对具身智能 / VLA / 模仿学习感兴趣的技术同行阅读，也作为我个人项目经历的沉淀。

---

## TL;DR

- 在 RoboDojo `stack_bowls` 任务上复现了 **ACT、Diffusion Policy、SmolVLA** 三种策略
- 所有模型任务成功率均约 **0%**，但失败模式完全不同
- 核心结论：**训练集（100 episode / 4~5 种布局）与评估集（85 种预定义布局）分布严重不匹配**，是失败主因，不是算法问题
- 过程中定位并修复了 **LeRobot、XPolicyLab、SmolVLA** 集成中的 3 个工程 bug
- 对 VLA 的理解：多任务预训练确实提升了视觉定位能力，但**物理接触精度**仍是数据驱动的瓶颈

---

## 1. 项目背景与目标

### 1.1 为什么做这个项目

我目标是找 **VLA（Vision-Language-Action）/ 具身智能** 方向的实习。招聘方除了看论文复现，更看重：

- 是否完整走通「**训练 → 部署 → 失败分析**」闭环
- 是否具备**定位真实瓶颈**的能力，而不是只会调参
- 是否有**工程落地能力**（修 bug、改 adapter、适配数据格式）

RoboDojo 是 2025 年提出的仿真-真机统一 benchmark，涵盖 42 个任务，评估 Generalization、Memory、Long-Horizon、Precision 四个维度。它的设计初衷就是测试 **generalist robot manipulation policies**，和我的目标高度契合。

### 1.2 为什么选 `stack_bowls`

`stack_bowls` 任务要求机械臂把三个碗叠在一起。我选它的原因：

- **多模态动作**：同一画面下存在多种合法策略（左手/右手、先夹哪个碗、不同轨迹）
- **视觉定位敏感**：碗的位置、颜色、材质变化大，考验模型对未见布局的泛化
- **长程操作**：「靠近 → 夹取 → 移动 → 叠放」需要稳定的动作序列
- **对 VLA 友好**：任务有明确语言描述（`Stack the three bowls together.`），适合测试 VLA 模型

### 1.3 项目目标

我一开始的目标不是「把成功率刷到 90%」，而是：

1. 在 RoboDojo 上完整跑通 ACT / Diffusion / SmolVLA 的训练与部署
2. 对比不同策略的失败模式
3. 定位失败根因，形成可复用的分析框架
4. 把踩坑和修复过程记录下来，体现工程能力

事后证明，第 3、4 点比第 1 点更有价值。

---

## 2. 实验环境与方法

### 2.1 硬件与软件

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 3090，24GB 显存 |
| 操作系统 | Ubuntu 24.04 |
| 仿真器 | Isaac Sim 5.1.0 |
| Policy 框架 | LeRobot v0.4.3（基于 JoyandAI 适配版） |
| 部署框架 | XPolicyLab（RoboDojo 官方 policy server/client） |
| Policy 环境 | `lerobot` conda env（Python 3.10） |
| 仿真环境 | `RoboDojo` conda env（Python 3.11） |

### 2.2 数据集

使用 RoboDojo v3.0 官方数据集：

- **路径**：`/media/endiyin/F/RoboDojo/data/RoboDojo_lerobot_v30_video`
- **格式**：LeRobot v3.0，视频格式（mp4），joint-only state/action
- **大小**：约 112 GB
- **总览**：35 个任务 × 100 trajectory = 3500 episode，约 186 万帧

其中 `stack_bowls` 任务：
- **episode 范围**：3000 ~ 3099（共 100 个 episode）
- **总帧数**：44874 帧
- **任务描述**：`Stack the three bowls together.`
- **数据特点**：包含 3~4 种不同位置、不同颜色的碗，动作分布多模态

### 2.3 评估方式

使用 XPolicyLab 提供的 closed-loop deploy pipeline：

- Policy 作为 server 加载 checkpoint，仿真环境作为 client 请求 action
- 每个 episode 最多 800 步
- 统计 `Success / Fail / Unstable` 数量
- 关键实验均跑 **15 episode** 取统计结果

---

## 3. 核心实验结果汇总

| 模型 / 配置 | 数据量 | 训练步数 | 最终 loss | 成功率 | 行为表现 | 失败原因 |
|---|---|---|---|---|---|---|
| ACT | 100 eps | 5K | ~0.000 | ~0% | 动到一半原地抽搐，不知如何继续 | 多模态动作被 MSE 平均，策略不收敛 |
| Diffusion 10K | 100 eps | 10K | - | ~0% | 更迷茫，经常找不到碗的位置 | 训练不足，未学会定位目标 |
| Diffusion 20K | 100 eps | 20K | 0.009 | ~0% | 能完成夹取→移动→尝试叠放，但夹取不稳、提前松抓 | 学到任务结构，但精度不足 |
| Diffusion 50K（开环） | 100 eps | 50K | - | ~0% | 流程完全正确，但夹爪夹得很浅，始终夹空气搬运空气 | 高层策略已学会，夹爪闭合精度不足 |
| Diffusion 50K（Closed-Loop） | 100 eps | 50K | - | ~0% | 流程仍正确，唯一一次夹住因抬升不够高推开底层碗 | Closed-loop 只修正执行流程，未提升夹取物理精度 |
| Diffusion 50K + image_transforms | 100 eps | 50K | 待补充 | ~0% | 仍然夹空气，无法精准定位碗的位置 | 训练集与评估集分布不匹配，增强无法弥补 |
| SmolVLA 100K（多任务） | 3500 eps (35 tasks) | 100K | 0.078 | ~0%（0/15） | 能大致定位到碗的位置，动作更自然，但夹爪夹得很浅 | 多任务学到通用夹取意图，但每个任务数据仍少，夹爪闭合精度未学好 |

**关键观察**：

- 随着训练步数增加（10K → 20K → 50K），Diffusion Policy 从「找不到碗」进步到「知道怎么做但夹不住」
- Closed-loop 和 image_transforms 对**执行流程**有改善，但对**成功率**没有实质提升
- SmolVLA 多任务训练让模型**定位能力明显更好**，但**夹取物理精度**仍是共性瓶颈
- 所有方法的成功率都卡在 **0%**，说明问题不在算法，而在数据

---

## 4. 失败模式深度分析

### 4.1 ACT：多模态动作被 MSE 平均

**现象**：机械臂动到一半后就在原地抽搐，看起来不知道接下来该做什么。

**根因**：
- `stack_bowls` 存在多种合法动作选择（左手/右手、先夹哪个碗）
- ACT 用 MSE 损失，本质上是多模态动作分布的平均
- 平均后的动作既不是「左手轨迹」，也不是「右手轨迹」，变成「四不像」

**人话解释**：就像同时听两个人指挥，一个说往左一个说往右，最后原地打转。

---

### 4.2 Diffusion 10K → 20K → 50K：从迷茫到「知道怎么做但做不好」

**Diffusion 10K**：
- 更迷茫，经常找不到碗的位置
- 夹取动作随机性强，没有稳定策略

**Diffusion 20K**：
- 能完成「靠近 → 夹取 → 移动 → 尝试叠放」完整流程
- 能看出叠放意图，但夹取不稳、提前松抓
- 对初始位置敏感：越接近训练分布，成功率越高

**Diffusion 50K（开环）**：
- 动作流程**完全正确**，像模像样
- 但夹爪夹碗时**夹得很浅**，始终在「夹空气、搬运空气」
- 即使没夹住，也会继续移动到叠放位置，说明只是**模仿动作流程**，没有真正理解「当前是否夹住了物体」

**人话解释**：20K 是「知道要叠碗但手笨」，50K 是「动作很标准但手里是空的」。

---

### 4.3 Diffusion 50K + Closed-Loop：执行流程改善，但物理精度没变

修改 deploy 脚本支持 `REPLAN_INTERVAL=1`，每执行 1 个 action 就重新规划。

**结果**：
- 跑了 10 多次，成功率仍接近 0%
- 唯一一次成功夹住碗并移动，但**抬升高度不够**，把底层碗推开了

**结论**：
- Closed-loop 能修正「执行时偏离 plan」的问题
- 但**不能补偿模型本身没学会的技能**：夹爪闭合深度不够、夹取后抬升高度不够

---

### 4.4 Diffusion 50K + image_transforms：增强无法弥补位置分布缺失

开启 brightness、contrast、saturation、hue、sharpness、affine 增强，训练 50K 步。

**结果**：
- 仍然频繁夹空气，无法精准定位碗的位置
- 相比无增强版本**没有明显改善**

**根因**：
- 评估时使用 **85 个预定义布局**（`stack_bowls_0.json` ~ `stack_bowls_84.json`）
- 训练集 100 个 episode 只覆盖约 **4~5 种**主要放置方式
- 评估布局中碗的 `category_idx` 分布很广（1~15），对应不同颜色/材质
- `image_transforms` 只能增强颜色/亮度，**无法让模型学会没见过的碗的位置**

---

### 4.5 SmolVLA 多任务：定位能力提升，但物理精度仍是瓶颈

使用 RoboDojo 全量 35 任务 / 3500 episode 训练 SmolVLA 100K 步。

**相比 Diffusion 的改进**：
- 夹爪能移动到目标碗的**大致上方**，不再隔着老远夹空气
- 动作更自然，接近人类示教
- 能看出模型在**理解任务意图**

**仍然失败的原因**：
- 夹爪闭合**仍然很浅**，无法真正夹起碗
- 多数 episode 中夹爪在碗表面滑过，夹空气后仍继续执行后续动作
- 3500 episode 对 35 个任务来说，每个任务只有 100 条数据
- 100K 步只相当于 **0.11 个 epoch**，模型远未收敛

**人话解释**：SmolVLA 更聪明，知道碗在哪里、该做什么，但「手劲」和「手感」还没练出来。

---

## 5. 关键发现：数据分布不匹配是根因

### 5.1 评估布局机制

我检查了 RoboDojo 的评估代码和场景布局文件，发现：

- 评估布局目录：`Assets/Eval_Layout/RoboDojo/arx_x5/0/`
- `stack_bowls` 任务共有 **85 个预定义布局**
- 每个 episode 按顺序选取一个布局文件
- 布局中碗的 `default_pos` 固定，`rotate_rand=false`
- 85 个布局中碗的 xy 位置覆盖桌面较大区域，`category_idx` 分布为 1,2,3,5,7,8,9,10,11,12,13,14,15

### 5.2 训练集与评估集的差距

| 维度 | 训练集 | 评估集 | 差距 |
|---|---|---|---|
| episode / 布局数量 | 100 个 episode | 85 个预定义布局 | 数量接近，但覆盖范围不对等 |
| 布局覆盖 | 约 4~5 种主要放置方式 | 85 种不同放置方式 | 训练集只覆盖评估集的 ~5% |
| 每个布局的演示数 | 可能多个 episode 来自同一布局 | 每个布局只有 1 次评估 | 模型没机会学习同一位置下的动作变化 |

### 5.3 类比理解

只给你看 **4~5 张不同桌子的照片**学做菜，考试时给你 **85 张完全不同的桌子**，还要你准确拿到指定杯子——即使对人类来说也很难。

### 5.4 RoboDojo 的设计意图

查阅 RoboDojo 论文后确认：**这不是 bug，而是设计意图**。

- 训练集设计为 35 任务 × 100 trajectory，测试策略能否泛化到评估时的分布变化
- 期望参赛者使用**多任务训练、预训练 VLA、数据增强、domain randomization** 等综合方法
- 官方 leaderboard 上：
  - 单任务 ACT 平均成功率仅 **0.32%**
  - 单任务 SmolVLA 平均成功率仅 **0.85%**
  - 最好的多任务 VLA 平均成功率也不到 **9%**

所以我们的实验结果（ACT ~0%、Diffusion ~0%、SmolVLA ~0%）**符合这个 benchmark 对单任务/小规模多任务模仿学习的预期**。

---

## 6. 工程踩坑与修复记录

### 6.1 LeRobot `episodes` 过滤 + `EpisodeAwareSampler` 的 IndexError

**现象**：
使用 `--dataset.episodes` 过滤子集训练 Diffusion Policy 时，启动后立刻崩溃：

```text
IndexError: Invalid key: 1634679 is out of bounds for size 44874
```

**根因**：
- Diffusion Policy 默认启用 `drop_n_last_frames=7`，`lerobot-train` 因此使用 `EpisodeAwareSampler`
- 该 sampler 返回的是整个数据集的**绝对帧索引**
- 但过滤后的 `hf_dataset` 只有 44874 行，导致 `__getitem__` 越界

**修复文件**：`/media/endiyin/F/lerobot/src/lerobot/datasets/lerobot_dataset.py`

**修复思路**：
在 `__getitem__` 中，如果加载了 episode 子集，先把 sampler 传进来的绝对索引映射到过滤数据集的相对索引，再访问 `hf_dataset`。

---

### 6.2 XPolicyLab adapter 硬编码 ACTPolicy

**现象**：
原 `model.py` 硬编码使用 `ACTPolicy.from_pretrained`，无法加载 Diffusion Policy checkpoint。

**修复文件**：`/media/endiyin/F/RoboDojo/XPolicyLab/policy/robodojo_act_lerobot/model.py`

**修复思路**：
改为从 checkpoint 中读取 `PreTrainedConfig`，根据 `config.type` 动态获取 policy 类（`act`、`diffusion`、`smolvla` 等），再调用 `policy_cls.from_pretrained`。现在同一个 adapter 可部署多种 policy。

---

### 6.3 SmolVLA `rename_map` 导致 `image_features` 不匹配

**现象**：
使用 `--rename_map` 将 RoboDojo 的相机名映射为 SmolVLA base 期望的 `camera1/camera2/camera3` 后，训练启动即崩溃：

```text
ValueError: All image features are missing from the batch.
```

**根因**：
- `make_policy()` 根据数据集特征填充 `cfg.input_features`（key 仍是 `cam_high/...`）
- `RenameObservationsProcessorStep` 只在运行时把 batch 里的 key 改名为 `camera1/2/3`
- SmolVLA 的 `prepare_images()` 读取的是 `policy.config.image_features`，该属性由 `input_features` 派生，仍包含旧 key
- 结果 batch 里是新 key，模型找的是旧 key，所有 image features 都匹配不上

**修复文件**：`/media/endiyin/F/lerobot/src/lerobot/policies/factory.py`

**修复思路**：
在 `make_policy()` 中，设置完 `cfg.input_features` 后，如果传入了 `rename_map`，把 `input_features` 的 key 也按 `rename_map` 重命名，保证模型期望的 key、preprocessor 输出的 key、batch 里实际的 key 三者一致。

---

## 7. 如果重来一次，我会怎么做

### 7.1 先分析评估集，再决定训练策略

我一开始直接拿 100 个 episode 训练，后来才发现评估有 85 个布局。如果重来：

1. 先解析评估布局文件，统计碗的位置、颜色分布
2. 检查训练集覆盖了哪些布局，缺失哪些
3. 优先补充缺失布局的数据，而不是随机增加数据

### 7.2 更早使用多任务训练或预训练 VLA

单任务 Diffusion 做到 50K 步才发现数据不够。如果重来：

- 更早尝试多任务 Diffusion Policy
- 或者直接上 SmolVLA / π0 等预训练 VLA
- 利用预训练模型的视觉先验，减少对同任务数据的依赖

### 7.3 把 closed-loop 和 image_transforms 定位为辅助手段

我花了较多时间测试 closed-loop 和 image_transforms，但它们都无法弥补**数据覆盖不足**这个根本问题。如果重来：

- 把它们作为「锦上添花」的手段，而不是「核心解决方案」
- 核心精力放在数据分布分析和模型选型上

### 7.4 更早做系统性对比

我一开始是「训一个、部署一个、看结果」，后来才整理对比表。如果重来：

- 更早建立统一的评估 pipeline 和对比表格
- 每个实验都记录 Success/Fail/Unstable 数量、行为视频、失败模式标签
- 这样分析根因会更高效

---

## 8. 对 VLA / 模仿学习的理解总结

### 8.1 模仿学习的本质是条件分布拟合

Behavior Cloning 学的是 `P(action | image, state)`。模型要根据视觉输入判断碗在哪里，然后输出对应的关节动作。**如果训练时某个位置从来没出现过，模型就没有任何依据输出正确的动作。**

### 8.2 数据覆盖决定上限

在这个项目里，我验证了：
- 增加训练步数（10K → 50K）能提升**任务结构理解**
- 使用 VLA 多任务预训练能提升**视觉定位能力**
- 但**物理接触精度**（夹爪闭合深度、闭合时机）仍然依赖**同任务数据的覆盖范围**

### 8.3 VLA 的优势与局限

**优势**：
- 多任务预训练让模型具备更好的视觉定位和任务理解能力
- 对未见过的布局，SmolVLA 至少能「找到碗」，而 Diffusion 经常「找不到」

**局限**：
- VLA 仍然需要足够的同任务数据来学习精细的物理操作
- 在数据极少的情况下，VLA 也不会比传统模仿学习好太多
- 100K 步 / 0.11 epoch 对 VLA 来说只是热身，远未收敛

### 8.4 工程能力比调参更重要

这个项目最大的收获不是「训出了多高的成功率」，而是：
- 完整走通了 LeRobot → XPolicyLab → Isaac Sim 的 pipeline
- 定位并修复了 3 个工程 bug
- 建立了系统的失败归因方法（数据分布 → 执行流程 → 模型能力）

在数据受限的真实场景中，**「知道为什么失败」比「盲目调参」更有价值**。

---

## 9. 附录：复现指南与关键文件索引

### 9.1 一键训练脚本

| 脚本 | 说明 |
|---|---|
| `/media/endiyin/F/RoboDojo/outputs/train/train_diffusion_stack_bowls.sh` | Diffusion 20K 步训练 |
| `/media/endiyin/F/RoboDojo/outputs/train/train_diffusion_stack_bowls_fast.sh` | Diffusion 50K 步高效配置 |
| `/media/endiyin/F/RoboDojo/outputs/train/test_smolvla_short.sh` | SmolVLA 1K 步短测试 |
| `/media/endiyin/F/RoboDojo/outputs/train/train_smolvla_multitask.sh` | SmolVLA 100K 步多任务训练 |

### 9.2 模型 Checkpoint

| 模型 | 路径 |
|---|---|
| Diffusion 20K | `/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls/checkpoints/020000/pretrained_model` |
| Diffusion 50K（fast） | `/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_fast/checkpoints/050000/pretrained_model` |
| Diffusion 50K + aug | `/media/endiyin/F/RoboDojo/outputs/train/diffusion_robodojo_stack_bowls_aug/checkpoints/050000/pretrained_model` |
| SmolVLA 100K | `/media/endiyin/F/RoboDojo/outputs/train/smolvla_robodojo_multitask/checkpoints/100000/pretrained_model` |

### 9.3 修改过的源码文件

| 文件 | 修复内容 |
|---|---|
| `/media/endiyin/F/lerobot/src/lerobot/datasets/lerobot_dataset.py` | 修复 episodes 过滤 + EpisodeAwareSampler 的 IndexError |
| `/media/endiyin/F/lerobot/src/lerobot/policies/factory.py` | 修复 SmolVLA rename_map 与 image_features 不匹配 |
| `/media/endiyin/F/RoboDojo/XPolicyLab/policy/robodojo_act_lerobot/model.py` | 支持自动识别 policy 类型 |

### 9.4 详细实验记录

- [`ROBODOJO_EXPERIMENTS.md`](./ROBODOJO_EXPERIMENTS.md)：完整实验流水账与部署观察
- [`DAILY_PROGRESS.md`](./DAILY_PROGRESS.md)：每日进度记录
- [`README.md`](./README.md)：项目总入口与快速开始指南

---

## 参考

- [RoboDojo 论文](https://arxiv.org/html/2607.04434v3)
- [LeRobot 官方仓库](https://github.com/huggingface/lerobot)
- [SmolVLA](https://huggingface.co/blog/smolvla)
- [ACT 论文](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)
