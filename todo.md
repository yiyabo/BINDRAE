# Stage-1 χ1 精度提升 TODO

## Phase 0：Baseline 确认（优先级：高，状态：已完成）
- [x] 复现 RemoveAllHs + Probes + 12D 类型编码的基准配置，在 batch_size=4、lr=1e-4 下重新完整训练一轮，确认 χ1≈71% 可稳定复现。

## Phase 1：χ1 rotamer 辅助头（优先级：最高，状态：已完成）
- [x] 设计 χ1 rotamer 离散 bin（如 g⁻ / t / g⁺ 等）及标签生成方式（从真实 χ1 角度映射到离散类别）。
- [x] 在 `TorsionHead` 中增加 χ1 rotamer 分类头，并将其接入总 loss（增加 rotamer 交叉熵损失）。
- [x] 更新训练配置，支持 rotamer loss 权重配置（如 `w_rotamer`）。
- [ ] 在 CASF-2016 上补充小规模 ablation 记录与复盘，整理 rotamer 头对 χ1 提升的实验结论。

## Phase 2：配体条件化增强（优先级：高，状态：已完成）
- [x] 设计多层 ligand conditioning 方案：在 IPA 的若干层之间插入轻量级 cross-attention（Protein ← LigandTokens），而不是只在前期做一次条件化。
- [x] 在 `LigandConditioner` / Stage-1 模型中实现多层配体条件化，确保显存占用和训练速度可接受。
- [ ] 在 CASF-2016 上系统评估多层 ligand conditioning 对整体 χ1 以及口袋残基 χ1 的影响，并整理文档。

## Phase 3：配体特征与侧链专门任务（优先级：中）
- [x] 设计并实现更丰富但低维的 ligand scalar 特征（如是否在环上、环大小 bucket、重原子度数 bucket、是否有杂原子邻居等），通过与 12D 类型向量拼接扩展为 20D，并在 `ligand_utils.py` / `LigandConditioner` / `dataset_ipa.py` 中打通。
- [ ] 预研 backbone-fixed 的侧链子任务：给定 backbone，仅预测侧链 torsion，用更强的 χ1/χ2 监督单独“打磨”侧链模块。
- [ ] 引入 Dunbrack 风格 rotamer 先验，作为 backbone-dependent rotamer 正则，约束预测不要偏离高概率 rotamer 区域。

## Phase 3b：Stage-1 伪-apo 扰动 denoising（优先级：中长期）
- [ ] 在 Stage1 里加一个小幅度的 “denoising autoencoder” 风格训练：人为对 holo torsion / backbone 加一点噪声（比如 χ1/χ2 偏一点、backbone 稍微扰动），视为“近 apo 状态”。
- [ ] 训练目标：在 ligand 条件下，把这个 “noised holo” 拉回原 holo，让 Stage1 在局部上学到一点“从邻近构象 → holo”的向量场。
- [ ] 将该局部向量场与 Stage2 的连续路径模型结合：Stage1 提供局部动力学先验，Stage2 通过 Flow Matching / SB 将这些局部信息串成长程 apo→holo 轨迹，保持与 RAE 中“Stage1 高保真重建 + Stage2 transport”的解耦精神一致。
