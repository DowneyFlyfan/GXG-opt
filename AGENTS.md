# Rules

- Baseline Optimizers are **Adam(W)** and **Muon**, which requires parameters tuning to get a good curve.

- All models must be of size in $[ 10M, 200M ]$

- When using Muon Optimizer, treat CNN parameter size as $(C_{out}, C_{in}*k_h*k_w / g)$

- if $r = \dfrac{C_{out}}{( C_{in} * k_h * k_w ) / g} > 4$ or $( C_{in} * k_h * k_w ) / g < 16$ or $C_{out} < 16$, use adam for this layer

- Weight Decay is only used for big models

- Muon is not applied on first layer of CNN

- Draw Metric-Steps Graph (png format) after every experiment together with baseline optimizers

- Plan should be created in current folder

- Be very careful to not go OOM (Out-Of-Memory)!!!

- Time limit for every experiment is 4 hours. Finish the last step and measure its metric if it hasn't finished yet.

# Server

- Use `ssh ABA` and only 1 GPU when it's fully available. Otherwise, run tasks locally
