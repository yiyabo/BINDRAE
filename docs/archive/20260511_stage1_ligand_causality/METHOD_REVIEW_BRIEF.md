# BINDRAE 方法论审核 Brief（供 GPT-5.4-pro 审阅）

> 目的：为外部高强度方法论审阅提供一份**聚焦当前真实状态**的说明文档，避免 reviewer 被历史文档、旧版 Stage-1、或过于超前的 Stage-2 蓝图误导。

---

## 0. 这份 Brief 要解决什么问题

BINDRAE 项目最初包含一个较完整的两阶段蓝图：

- **Stage-1**：学习一个 ligand-conditioned holo structural prior / decoder；
- **Stage-2**：在混合状态空间上学习 apo→holo 的连续桥流 / 路径生成。

但项目当前的**实际研发状态**已经收敛为：

1. **当前主线是 Stage-1 封板**，不是 Stage-2 落地；
2. Stage-1 的定位也已进一步明确为：
   - **ligand-conditioned pocket structural prior**，
   - 更具体地说，是一个偏向 **binding pocket / local sidechain geometry** 的先验模块；
3. 当前真正卡住的问题不是“模型能不能训练稳定”，而是：
   - **Stage-1 产生的 pocket prior 还不够强**；
   - 关键指标 `pocket_chi1_acc` 目前最好约在 **0.3657**，我们认为这还不足以作为非常强的 Stage-2 structural prior。

因此，这份 Brief 的目标不是重新介绍整个项目，而是把 reviewer 需要判断的事情收束为：

> **Stage-1 当前的方法定位是否合理？当前指标算不算偏低？下一步优化最该优先什么？项目文档和方法叙事是否需要调整？**

---

## 1. 当前项目状态（必须以此为准）

### 1.1 当前主线

- **主线：Stage-1 优化与封板**
- **非主线：Stage-2 理论设计已较完整，但暂未进入主实验推进**

### 1.2 当前最重要的真实目标

我们当前并不把 Stage-1 当作一个“全局 holo 结构重建器”来优化，而是更具体地把它视为：

> **给定 apo backbone + ligand pose 时，学习一个 ligand-conditioned、pocket-sensitive、可用于 Stage-2 的局部结构先验。**

换句话说，当前最看重的是：

- pocket 区域 sidechain / local geometry 恢复质量；
- 作为 Stage-2 终点先验 / path prior 的可用性；
- 而不是单纯把全局 `val_total_loss` 压到最低。

### 1.3 当前关于 Stage-2 的立场

Stage-2 仍然是项目总体方法论的重要组成部分，但**当前不会优先推进 Stage-2 实现**。原因是：

- Stage-2 质量高度依赖 Stage-1 prior 强度；
- 如果 Stage-1 的 pocket prior 仍偏弱，则过早进入 Stage-2 会把问题耦合、放大，并模糊归因；
- 因此当前策略是：
  - **先把 Stage-1 prior 做强并封板**；
  - 再把它固定为 Stage-2 的 decoder / prior / geometry toolkit。

---

## 2. 相对原始方案，我们已经做过哪些关键修改

这里的“原始方案”指：

- 文档中的最初 Stage-1 / Stage-2 方法蓝图；
- 以及项目早期更偏理论驱动、日志与选模相对粗糙的训练流程。

### 2.1 修改一：从“总 loss 驱动”转向“任务关键指标驱动”

我们已经确认：如果继续只用 `val_total_loss` 选模，会掩盖 pocket 目标上的改善。

因此已完成的改动包括：

- 新增结构化 epoch 日志：`metrics.jsonl`
- 新增多种 checkpoint：
  - `best_model.pt`
  - `latest_model.pt`
  - 可选 `epoch_{k}.pt`
- 新增可配置 `selection_metric`
- 新增并显式记录 `pocket_chi1_acc`

这一步的意义是：

> 训练现在不再只是“跑完看总 loss”，而是能明确判断 **模型是否真的在我们关心的 pocket prior 目标上进步**。

### 2.2 修改二：Stage-1 的评估重点已经发生改变

当前我们更关注：

- `pocket_chi1_acc`
- `chi1_acc`
- `val_fape`
- 稳定性 / 过拟合趋势

而不再把：

- `val_total_loss` 的最小值

视为唯一 best checkpoint 标准。

### 2.3 修改三：开始系统性测试 loss 对齐问题

当前已经做过或正在做的优化链路：

- **E1**：补日志、补 checkpoint 保存
- **E2**：支持可配置 `selection_metric`，并实际切到 `pocket_chi1_acc` 选模
- **E3**：测试 `w_fape: 1.0 → 0.5`，评估是否因为 FAPE 主导导致 pocket χ 学不起来

这代表当前项目已经从“模型搭起来能跑”进入：

> **有明确假设、按单变量逐步验证的优化阶段。**

### 2.4 修改四：Stage-1 定位已从“泛 holo decoder”收紧到“pocket prior”

这个修改不一定完全体现在旧文档里，但已经是当前实际研发共识：

- Stage-1 不追求独立完成完整的 induced-fit 问题；
- Stage-1 的价值在于：
  - 提供 **ligand-conditioned local pocket prior**；
  - 提供 FK / FAPE / clash / torsion 这套 geometry toolkit；
  - 在 Stage-2 中充当 endpoint prior / 后半程 path prior。

---

## 3. 当前 Stage-1 方法定义（以当前实现为准）

### 3.1 任务定义

输入：

- protein sequence 的 frozen ESM 表征；
- apo backbone（N / Cα / C）及由其构造的 backbone frame；
- ligand 3D pose 与 ligand token 特征；
- apo–holo 配对训练数据中的 holo sidechain / holo target structure 作为监督。

输出：

- per-residue sidechain χ torsion（当前核心是 χ1–χ4）
- 结合 backbone frame / FK 得到的全原子几何重建

训练目标：

- 在 ligand 条件下恢复 holo-like local geometry；
- 更具体地，提高 pocket 区域 sidechain 恢复质量；
- 同时保持 backbone-local geometry / atom-level physical plausibility。

### 3.2 当前 loss 结构（代码实现层面）

当前 Stage-1 训练总损失为：

\[
L_{total} = w_{fape} \cdot L_{fape} + w_{chi} \cdot L_{chi} + w_{clash} \cdot L_{clash}
\]

当前实现默认值：

- `w_fape = 1.0`
- `w_chi = 1.0`
- `w_clash = 0.1`

其中：

- `L_chi`：wrap-aware torsion sin/cos loss
- `L_fape`：基于 N/Cα/C 构造 frame 的 FAPE
- `L_clash`：atom14 上的 clash penalty

此外还有两个重要训练机制：

1. **Pocket warmup**
   - `w_res` 在训练初期从 0.1 逐步 warmup 到真实 pocket 权重；
2. **Ligand gate warmup**
   - 配体条件化分支逐步打开，避免训练早期被 ligand signal 干扰。

### 3.3 当前 pocket weighting 的实现语义

在数据集里，`w_res` 由 apo CA 与 ligand 的最近距离经 logistic/sigmoid 软权重得到：

\[
w_{res} = \frac{1}{1 + \exp((d_{min} - d_0)/\tau)}
\]

当前默认：

- `d0 = 6.0 Å`
- `tau = 1.0`

因此：

- 训练时 `w_res` 是**软权重**；
- 验证时定义 pocket 指标时使用 `w_res > 0.5` 作为 hard pocket mask；
- 这大致对应“apo CA 到 ligand 最近距离 < 6 Å”的口袋区域。

---

## 4. 当前关键指标定义（请 reviewer 以此解释结果）

### 4.1 `chi1_acc`

定义：

- 仅在 χ1 有定义的残基上计算；
- 取预测 χ1 与真值 χ1 的 wrap 后角度差；
- 若 `|Δχ1| < 20°`，记为命中；
- `chi1_acc = hits / valid_residues`

这本质上是标准的 χ1 rotamer recovery / torsion hit 指标。

### 4.2 `pocket_chi1_acc`

定义：

- 与 `chi1_acc` 相同；
- 但只在满足以下条件的残基上计算：
  - χ1 有定义；
  - `w_res > 0.5`；
  - `node_mask == True`

因此：

> `pocket_chi1_acc` 是 **pocket residues 上 χ1 在 20° 阈值内命中的比例**。

### 4.3 `val_fape`

定义：

- 基于真值局部 frame 计算 frame-aligned point error；
- 当前实现是局部 frame 下 atom14 坐标误差，经 clamp 后做平均；
- 主要反映 backbone / local geometry 是否稳定。

### 4.4 `val_chi`

重要说明：

- 当前 `val_chi` 对应的是训练 loss 中的 χ torsion loss；
- 该 loss 已使用 pocket-weighted `w_res`（含 warmup 逻辑）进行加权；
- 因此它并不是一个完全“全局平均”的 χ 指标，而是已经对 pocket 区域更敏感。

### 4.5 为什么当前最重要的是 `pocket_chi1_acc`

因为当前 Stage-1 的角色不是通用 sidechain packer，而是：

> **供 Stage-2 使用的 ligand-conditioned pocket structural prior**。

因此与 Stage-2 成功更强相关的，不是“总 loss 最低”，而是：

- pocket 区域 χ1 是否稳定恢复；
- pocket prior 是否足够强；
- 是否能为 Stage-2 提供清晰、可靠的局部几何锚点。

---

## 5. 当前实验快照（截至本次审阅）

### 5.1 数据与训练设定概况

当前 Stage-1 使用的是自建 apo–holo–ligand triplet 数据：

- 原始候选约来自 51.6 万 PDB 记录筛选；
- 最终保留约 9.1 万高质量样本；
- 当前一版训练/验证规模约为：
  - train: **65,622**
  - val: **2,863**

当前稳定训练配置（近期主线）大致为：

- `model_size = medium`
- 8×A100 40GB
- `batch_size = 8` / rank（effective global batch 64）
- `max_n_res = 1600`
- `residue_budget = 1600`
- `pocket_warmup_steps = 8000`
- `ligand_gate_warmup_steps = 8000`
- LR scheduler = plateau
- mixed precision = bf16
- early stop patience = 20

### 5.2 E2：selection metric 切换实验（已完成）

实验目标：

- 验证之前 best checkpoint 总停在 very early epoch，是否是因为用 `total val loss` 选模掩盖了 pocket 改善。

改动：

- 保持训练主配置不变；
- 将 `selection_metric` 切换为 `pocket_chi1_acc`。

结果：

- E2 best checkpoint：`epoch 11`
- **best `pocket_chi1_acc = 0.3657`**

解释：

- 这证明“只看 total loss 选 best”确实会错过 pocket 改善；
- 但也暴露了更深层问题：
  - `pocket_chi1_acc` 虽然能在早中期提升，
  - 但总体提升仍偏弱，且较快进入平台区。

### 5.3 E3：降低 FAPE 权重实验（进行中/阶段性观察）

实验目标：

- 检查当前 loss 中 FAPE 是否在数值尺度上主导优化方向，导致 pocket χ 改善受限。

改动：

- 在 E2 基础上仅修改：
  - `w_fape: 1.0 -> 0.5`
- 仍用 `pocket_chi1_acc` 选模。

当前阶段性观察（运行中快照）：

- 训练稳定；
- 但截至中早期观测，**尚未证明 E3 优于 E2**；
- 当前最关键的判断标准仍是：
  - `pocket_chi1_acc` 能否明显超过 E2 的 **0.3657**。

### 5.4 当前对结果的总体判断

我们目前的判断不是“模型训坏了”，而是：

1. **训练工程是健康的**
   - 训练稳定
   - 指标可观测
   - checkpoint 逻辑清晰
   - 多卡训练正常

2. **Stage-1 确实学到了东西，但 pocket prior 还不够强**
   - `pocket_chi1_acc ≈ 0.35–0.37`
   - 我们认为这说明模型已具备一定 pocket sensitivity
   - 但还未达到“强 prior / 可放心封板”的程度

3. **当前最核心瓶颈是目标函数没有足够对准 pocket prior**
   - 不是简单的 training instability
   - 更像是：
     - FAPE / geometry term 与 pocket χ 目标之间存在优化竞争；
     - 模型学到一些通用结构恢复，但没有足够集中地学到 pocket sidechain recovery。

---

## 6. 我们当前如何理解“这个结果高还是低”

### 6.1 不能直接对标经典 fixed-backbone sidechain packing

文献中，经典或现代 sidechain packing 方法在 **native / near-native backbone** 条件下往往能获得很高的 χ1 recovery。

例如 SCWRL4 在更宽松阈值（40°）下可报告很高的 χ1/χ1+2 恢复率。这说明：

> 在 fixed-backbone、特别是实验 backbone 输入下，sidechain packing 是一个相对更容易的问题。

而我们的 setting 更难：

- apo 而非 holo backbone；
- ligand-conditioned；
- pocket-focused；
- 目标是服务于 apo→holo prior，而非单纯全局 repacking。

### 6.2 但即便考虑任务更难，我们仍认为当前结果偏低

我们当前对 `pocket_chi1_acc ≈ 0.35–0.37` 的判断是：

- **不是离谱地差**；
- 但**对一个想服务 Stage-2 的 pocket structural prior 来说，仍偏弱**。

当前内部的工程判断区间为：

- `< 0.35`：信号较弱，不建议封板
- `0.35 – 0.40`：已有可用性，但 prior 强度不足
- `0.40 – 0.45`：开始接近像样的 Stage-1 prior
- `> 0.45`：较强 prior
- `> 0.50`：非常理想，具备较强说服力

因此当前阶段：

> 我们更倾向于把 Stage-1 定义为“稳定、可用、有方向性，但还没封板”。

---

## 7. 当前最需要 reviewer 审核和挑战的点

以下是我们最希望 GPT-5.4-pro 严格审查的内容。

### 7.1 Stage-1 的当前定位是否合理？

当前定位是：

- 不把 Stage-1 当作完整 induced-fit 解算器；
- 而是把它定义为：
  - ligand-conditioned pocket structural prior
  - Stage-2 的 decoder / prior / geometry toolkit

我们希望 reviewer 判断：

- 这种任务分解是否合理；
- 是否应该进一步收紧 Stage-1 目标；
- 或者相反，是否 Stage-1 现在承担得过少、导致 Stage-2 负担过重。

### 7.2 当前 `pocket_chi1_acc` 作为主指标是否合适？

我们当前认为：

- `val_total_loss` 不足以代表 Stage-1 对 Stage-2 的价值；
- `pocket_chi1_acc` 更能反映 Stage-1 prior 强度。

希望 reviewer 判断：

- 这一判断是否成立；
- 是否应当再加入其他更合适的主指标 / 组合指标；
- 如是否需要更强调 pocket RMSD、局部 contact consistency、按 residue type 分解后的恢复率等。

### 7.3 当前 loss 结构是否与目标不匹配？

当前 loss：

\[
L = w_{fape} L_{fape} + w_{chi} L_{chi} + w_{clash} L_{clash}
\]

我们的怀疑是：

- `L_fape` 数值尺度较大，可能主导优化；
- 从而让模型更偏向 backbone/local frame consistency；
- 而不是 pocket χ 恢复。

希望 reviewer 判断：

- 这个诊断是否合理；
- 当前更优的方向应该是：
  1. 继续调 `w_fape`
  2. 做 FAPE warmup
  3. 直接做 **pocket χ 增权**
  4. 改 metric 组合
  5. 改 loss 结构而非权重

### 7.4 下一步优化顺序是否正确？

当前我们的直觉优先级是：

1. 等 E3 完整结束并与 E2 对照；
2. 若 E3 不能明显超过 E2，下一步优先：
   - **E5：pocket chi 定向增权**
3. 再考虑：
   - **E4：FAPE warmup**

我们希望 reviewer 判断：

- 这个优先级是否合理；
- 是否存在更优但我们尚未明确考虑的中等改动；
- 哪些优化最符合“单变量、可解释、能快速证伪”的原则。

### 7.5 文档叙事是否应该调整？

当前项目文档中，Stage-2 理论蓝图较完整；但真实工作状态仍然以 Stage-1 封板为主。

希望 reviewer 判断：

- 当前方法论文档是否需要更明确地区分：
  - “最终方法蓝图”
  - “当前真实研发阶段”
- 是否应该把 Stage-1 的叙事从“generic holo decoder”进一步改写为“pocket prior module”。

---

## 8. 当前我们自己的主观判断（供 reviewer 参考，也欢迎推翻）

### 8.1 我们当前认为正确的部分

1. **两阶段方法论总体是合理的**
   - 先学 structural prior
   - 再学 apo→holo path / bridge flow

2. **Stage-1 当前聚焦 pocket prior 是合理的**
   - 这是最符合 Stage-2 需求的切分方式

3. **当前训练工程方向是正确的**
   - 先补日志、选模、checkpoint
   - 再做 loss 对齐实验

### 8.2 我们当前最不确定的部分

1. `pocket_chi1_acc` 作为主指标是否足够
2. `FAPE` 是否确实是当前最大的优化干扰项
3. 下一步最优动作究竟是：
   - pocket χ 增权
   - FAPE warmup
   - 还是更改 loss 结构/metric 设计
4. 当前 Stage-1 最现实的“封板阈值”应该设在哪里

---

## 9. 附：建议 reviewer 一起参考的项目文档

建议与本 Brief 一起阅读：

1. `docs/STAGE1_SUMMARY.md`
   - 旧版 Stage-1 总结，主要用于理解历史脉络（注意其中是 legacy CASF holo-only 路线）
2. `docs/stage1_optimize.md`
   - 旧版优化想法与方法调研（其中不少内容偏早期设想，需结合本 Brief 理解）
3. `docs/STAGE1_PIPELINE.md`
   - 旧版 Stage-1 pipeline 说明（legacy）
4. `docs/理论/Stage2理论与指导.md`
   - 当前 Stage-2 理论蓝图与设计原则

> 重要提醒：
> - `STAGE1_SUMMARY.md` 与 `STAGE1_PIPELINE.md` 目前更多代表 **legacy CASF holo-only Stage-1**；
> - 本 Brief 才是当前 Stage-1 实验状态与优化目标的最新收束版本。

---

## 10. 希望 reviewer 最终回答的问题（可直接逐条回应）

请基于以上信息，重点回答：

1. **Stage-1 当前被定义为 ligand-conditioned pocket structural prior，这个定位是否合理？**
2. **`pocket_chi1_acc` 是否是当前最合适的主指标？如果不是，应该换成什么或增加什么？**
3. **当前 best `pocket_chi1_acc ≈ 0.3657`，在这个任务设定下你认为算什么水平？是否偏低？**
4. **当前 loss 结构是否与目标存在错配？你最怀疑哪一项？**
5. **若只能优先做一项下一步优化，你建议选什么？为什么？**
6. **你是否建议把项目叙事进一步改成“Stage-1 = pocket prior module, Stage-2 = continuous apo→holo bridge”这一更明确的版本？**
7. **如果你要给当前 Stage-1 设一个“可封板”的标准，你会怎么定义？**

---

## 11. 一句话总结

> BINDRAE 当前不是在争论“大方法是否成立”，而是在解决一个更具体、更现实的问题：
> **如何把 Stage-1 从“已经稳定、已有信号的 pocket-aware predictor”，推进到“足够强、足够可信、可供 Stage-2 使用的 ligand-conditioned pocket structural prior”。**
