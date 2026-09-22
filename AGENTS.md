# Rules

- Baseline Optimizers are **Adam(W)**, **Muon** and **Muown**, which requires parameters tuning to get a good curve.

- When using Matrix-Based Optimizer, treat CNN parameter size as $(C_{out}, C_{in}*k_h*k_w / g)$. And don't apply it to first layer of CNN; Non-Matrix Parameters are all updated using Adam

- In any case,  if $r = \dfrac{C_{out}}{( C_{in} * k_h * k_w ) / g} > 4$ or $( C_{in} * k_h * k_w ) / g < 16$ or $C_{out} < 16$, use adam for this layer

- Weight Decay is only used for big models (bigger than 200M)

- Draw Metric-Steps Graph(Step, not Epoch!!!) and Metric-Time Graph (png format) together with baseline optimizers for every final optimizer result. (Parameter Tuning Data should be reserved, but image is unnecessary)

- Train all NLP models for 5 epochs

- Tune batch size so that it almost saturates the GPU memory. Use gradient accumulation to increase batch size.

- Equivalent Batch size for every optimizer should be the same; It should be 64 for NLP Task

# Metric, Model & Dataset

- Train from scratch

- Metric for NLP is perplexity; for Classfication is accuracy

- Model1: GPT2-12x512; Dataset: WikiText-103-Raw: 100,000,000 UTF-8 byte tokens; Evaluation: 10,000,000 UTF-8 byte tokens

- Model2: Qwen3-0.6B; Dataset: `/home/yufan/New_Optimizer/.cache/Fineweb_Edu_2B` on ABA server

- Model3: ViT-S/16; Dataset: ImageNet-100

# Server

- Run Task on a ABA A100 (Use `ssh ABA`) if it's available. Otherwise, run it on local GPU.
