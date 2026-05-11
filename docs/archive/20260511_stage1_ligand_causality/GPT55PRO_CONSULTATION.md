# 咨询 GPT-5.5 Pro：BINDRAE Stage-1 Ligand-Causal Posterior 优化方向

## 项目背景

BINDRAE 是一个蛋白质配体结合构象预测模型，采用两阶段架构：

- **Stage-1**：给定 apo（未结合）蛋白骨架 + 配体3D坐标 → 预测 holo（结合态）侧链扭转角（chi1-chi4）
- **Stage-2**：Torsion Flow Model，用 Stage-1 的 posterior 作为 soft prior/guidance 来生成侧链构象

**核心目标**：让 Stage-1 的 χ1 rotamer posterior 变成 **ligand-causal** 的——即模型预测的旋转异构体状态确实是"因为看到了正确的配体"才做出的，而不是单纯记住了骨架偏好。

## 当前模型架构

```
ESM-2(frozen) → Adapter(1280→384) → LigandConditioner(Cross-Attn + FiLM) → EdgeEmbed → FlashIPA(8 blocks) → s_geo(384-dim)
                                                                                                                    ↓
                                                                                                            TorsionHead → chi1-4 sin/cos
                                                                                                            CandidateChi1Scorer → 3-class rotamer logits
```

### LigandConditioner 细节
- 配体表示：原子类型 embedding + 3D坐标 → MLP/EnhancedEncoder(RBF距离+自注意力) → d_lig=64
- Cross-Attention：蛋白 node(384) attend to 配体 tokens(64)，8 heads
- FiLM 调制：gamma/beta MLP，门控 warmup
- 配体信息在 IPA 之前注入，IPA 内部也有 ligand conditioner 的 re-injection

### CandidateChi1Scorer（当前实验）
```python
class CandidateChi1Scorer(nn.Module):
    def __init__(self, c_s=384, c_hidden=128):
        self.candidate_embedding = nn.Parameter(torch.zeros(3, c_s))  # 3个rotamer候选
        self.scorer = nn.Sequential(
            LayerNorm(c_s * 2),
            Linear(c_s * 2, c_hidden), GELU, Dropout,
            Linear(c_hidden, 1)
        )
    
    def forward(self, s):  # s = s_geo [B, N, 384]
        # 每个残基与3个候选embedding拼接后打分
        s_expanded = s.unsqueeze(2).expand(B, N, 3, C)
        candidate = self.candidate_embedding.expand(B, N, 3, C)
        pair = cat([s_expanded, candidate], dim=-1)  # [B, N, 3, 768]
        return self.scorer(pair).squeeze(-1)  # [B, N, 3] logits
```

### Ligand Contrastive Loss
```python
def ligand_contrastive_chi1_loss(correct_logits, decoy_logits, true_chi1, chi1_mask, margin=0.5):
    """
    correct_logits: 用正确配体得到的 candidate logits
    decoy_logits: 用 batch-shuffled 配体（同batch内其他样本的配体）得到的 logits
    对 gold rotamer bin 的 logit 施加 margin loss：
    loss = ReLU(margin - (correct_gold_logit - decoy_gold_logit))
    """
```

训练时做两次 forward：一次用正确配体，一次用 batch 内打乱的配体作为 decoy。

## 实验历史与结果

| 实验 | val_chi1_rotamer_acc | 说明 |
|------|---------------------|------|
| Frozen posterior head only | 0.495 | 冻结backbone，只训练3-class MLP head |
| + Unfreeze ligand conditioner | 0.502 | 解冻配体条件化模块 |
| + Candidate scorer (1 epoch, LR=3e-5) | 0.508 | 当前candidate架构，LR太低 |
| + Candidate scorer (10x LR=3e-4, 进行中) | ? | 正在4-GPU训练 |

**Ligand-ablation 诊断结果（frozen head checkpoint）：**
- 正确配体 vs 无配体 vs 打乱配体 → rotamer accuracy 几乎无差异
- 结论：posterior 不是 ligand-causal 的，配体信息没有有效传递到 rotamer 预测

**关键观察：**
- val_candidate_chi1 CE loss ≈ 1.04（random = ln(3) ≈ 1.10），scorer 几乎没学到
- 训练 candidate loss 降到 0.64-0.86，但验证仍接近 random → 泛化困难
- Backbone 冻结，只有 650K trainable params（ligand conditioner + candidate scorer）

## 我的核心困惑

### 1. 信息瓶颈问题
CandidateChi1Scorer 的输入是 `s_geo`（IPA 输出的 384-dim node features）。如果 LigandConditioner + IPA 没有把足够的配体信息编码进 `s_geo`，那 scorer 再怎么训也学不到 ligand-causal 的 pattern。

**问题**：如何判断瓶颈在哪里？是 ligand conditioner 注入不够，还是 IPA 传播不够，还是 scorer 本身表达力不够？

### 2. Scorer 架构是否合理
当前 scorer 只看 `s_geo`（已经是 ligand-conditioned 的 node feature），用 candidate embedding 做 bilinear-style scoring。它没有：
- 显式的残基-配体距离特征
- 残基-配体原子级别的 attention
- 残基类型信息（aatype）
- 邻居残基的上下文

**问题**：对于 ligand-causal rotamer prediction，scorer 应该看什么信息？纯靠 s_geo 够吗？

### 3. Contrastive Loss 设计
当前用 batch-shuffled ligand 作为负样本。问题：
- 不同蛋白的配体差异很大，模型可能只学会了"这个配体形状不匹配这个口袋"而不是"这个配体导致了这个rotamer变化"
- Margin loss 在 logit 空间操作，可能不够直接

**问题**：有没有更好的方式强制 ligand causality？比如：
- 用同一蛋白的不同配体作为 hard negative？
- 直接在 rotamer accuracy 上做对比（correct ligand acc > no ligand acc）？
- 用 information bottleneck / mutual information 的方法？

### 4. 根本性问题：这个任务可行吗？
从物理角度看，配体结合确实会改变口袋残基的 rotamer 分布。但：
- 很多残基的 rotamer 主要由骨架 phi/psi 决定（Dunbrack library），配体只影响少数口袋残基
- 数据集中 apo→holo 的 chi1 变化可能很小（很多残基本来就在正确的 rotamer bin）
- 如果只有 5-10% 的残基真正被配体"翻转"了 rotamer，信号非常稀疏

**问题**：在这种稀疏信号下，什么样的训练策略最有效？是否应该只关注 pocket 残基而不是全局？

## 我想要的建议

1. **架构方向**：如果当前 candidate scorer 路线走不通，下一步最值得尝试的架构是什么？
2. **Loss 设计**：有没有比 margin contrastive 更好的方式来强制 ligand causality？
3. **训练策略**：考虑到信号稀疏性，有什么 curriculum / hard mining / focal 策略推荐？
4. **诊断方法**：如何快速判断瓶颈在 ligand conditioner 还是 scorer？
5. **文献参考**：在 ligand-conditioned side-chain prediction 或 structure-conditioned rotamer prediction 领域，有没有相关的 SOTA 方法可以借鉴？

## 约束条件
- Stage-1 backbone（ESM adapter + IPA）已经训练好，不想从头重训（太贵）
- 可以 unfreeze ligand conditioner 和最后几个 IPA blocks
- 配体表示是 3D 坐标 + 原子类型（不是 SMILES/graph）
- 训练数据约 65K 样本（apo-holo triplets），验证 2.8K
- 计算资源：4×A100 40GB，可以跑 24h
