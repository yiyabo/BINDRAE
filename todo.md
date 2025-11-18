# Stage-1 χ1 精度提升 TODO

## Phase 0：Baseline 确认（优先级：高）
- [ ] 复现 RemoveAllHs + Probes + 12D 类型编码的基准配置，在 batch_size=4、lr=1e-4 下重新完整训练一轮，确认 χ1≈71% 是否可稳定复现。

## Phase 1：χ1 rotamer 辅助头（优先级：最高）
- [ ] 设计 χ1 rotamer 离散 bin（如 g⁻ / t / g⁺ 等）及标签生成方式（从真实 χ1 角度映射到离散类别）。
- [ ] 在 `TorsionHead` 中增加 χ1 rotamer 分类头，并将其接入总 loss（增加 rotamer 交叉熵损失）。
- [ ] 更新 `train_stage1.py`，支持 rotamer loss 开关与权重配置（例如 `--rotamer_loss_weight`）。
- [ ] 在 CASF-2016 上跑一轮小规模对比实验：
  - 对照组：无 rotamer 头（当前版本）。
  - 实验组：加入 rotamer 头，设置不同权重（如 0.1 / 0.2）。
  - 记录 χ1、Val loss、收敛速度，评估 rotamer 头对 χ1 的提升幅度。

## Phase 2：配体条件化增强（优先级：高）
- [ ] 设计多层 ligand conditioning 方案：在 IPA 的若干层之间插入轻量级 cross-attention（Protein ← LigandTokens），而不是只在前期做一次条件化。
- [ ] 在 `LigandConditioner` / Stage-1 模型中实现多层配体条件化，确保显存占用和训练速度可接受。
- [ ] 在 CASF-2016 上评估多层 ligand conditioning 对整体 χ1 以及口袋残基 χ1 的影响。

## Phase 3：配体特征与侧链专门任务（优先级：中）
- [ ] 设计更丰富但低维的 ligand scalar 特征（如 hybridization、是否在环上、RDKit partial charge、HBD/HBA 计数等），通过小 MLP 投影后与 12D 类型向量拼接。
- [ ] 预研 backbone-fixed 的侧链子任务：给定 backbone，仅预测侧链 torsion，用更强的 χ1/χ2 监督单独“打磨”侧链模块。
- [ ] 引入 Dunbrack 风格 rotamer 先验，作为 backbone-dependent rotamer 正则，约束预测不要偏离高概率 rotamer 区域。

## Phase 4：数据与长期扩展（优先级：中长期）
- [ ] 评估将数据集从 CASF-2016 扩展到更大规模复合物库（如 PDBbind 等）的可行性，统一通过 `prepare_ligands.py` 与一致性检查管线处理。
- [ ] 分析“口袋残基 χ1 vs 全局 χ1”，根据口袋 χ1 的实际水平决定后续是否重点针对 pocket 区域优化。
- [ ] 在 Stage-2（连续路径学习）到来后，设计与 Stage-1 兼容的中间状态监督或正则，使构象路径更加光滑、物理合理。
