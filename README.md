# Ideas

The repository-native implementation of the Kronecker GGN baseline and its signed low-rank relative-residual correction is documented in [KRONECKER_GGN_README.md](KRONECKER_GGN_README.md).

- [x] Adaptive Dropout Rate for each layer

- [x] Alternative Update: Adam -> GN -> Adam (With momentum saved) -> GN -> ...

- [ ] Adaptive Layer Frozen in later section of training

- [x] Low-Rank GN

# Steepest Descent with Effective Rank Constraint

- Result: Between Muon and Muown

- [] Tune lr and effective normalized rank

![Effective-Rank-Half metric versus steps](results/nlp/effective_rank_half_lr000125_b8_a6_final_metric_steps.png)

The completed five-epoch GPT2-12x512 perplexity comparison, including the
joint-Newton effective-rank continuation, is shown below.

![Five-epoch GPT2-12x512 effective-rank comparison](results/nlp/gpt2_ppl_aba_a100_ppl_formal_b8_a6_joint_newton_four_way_steps.png)

# Hybrid Stiefel-Muon

![Hybrid Stiefel-Muon metric versus steps](results/nlp/hybrid_stiefel_muon_b12_a4_mu0006_st003_formal_metric_steps.png)

# Low-Rank Gauss Newton

- Low-Rank GN is helpful in early stage of training.

![image](./results/nlp/nystrom_adam_formal_stage1_ns15e6_adam15e5_b12_a4_metric_steps.png)

# Claims

 > All ranks here are stable ranks ($||A||_* / ||A||_F$)

## FFT 

- Shallow layers have more high frequency components;

- Deep layers have more low frequency components;

- Frequencies are usually very concetrated and sparse -> Fixed set of frequency basis spans useful gradient subspaces ???

## Wavelet-Transform

## Original Weights

- Gradient Ranks: Low -> Sudden high -> Stable

- Activation Matrices ranks are all-the-way low

- Directions with small singular values matter <- Delete them degrades performance greatly
