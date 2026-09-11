# Ideas

The repository-native implementation of the Kronecker GGN baseline and its signed low-rank relative-residual correction is documented in [KRONECKER_GGN_README.md](KRONECKER_GGN_README.md).

- [x] Adaptive Dropout Rate for each layer

- [x] Alternative Update: Adam -> GN -> Adam (With momentum saved) -> GN -> ...

- [ ] Adaptive Layer Frozen in later section of training

- [x] Low-Rank GN

# Effective-Rank-Half certified update

- Between Muon and Muown

- [] Tune lr, effective normalized rank

![Effective-Rank-Half metric versus steps](results/nlp/effective_rank_half_lr000125_b8_a6_final_metric_steps.png)

# Hybrid Stiefel-Muon

![Hybrid Stiefel-Muon metric versus steps](results/nlp/hybrid_stiefel_muon_b12_a4_mu0006_st003_formal_metric_steps.png)

# Low-Rank Gauss Newton

- Low-Rank GN is helpful in early stage of training.

![image](./results/nlp/nystrom_adam_formal_stage1_ns15e6_adam15e5_b12_a4_metric_steps.png)
