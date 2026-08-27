# Muon Computer Vision Baseline Diagnosis

## Question

Determine whether Muon was applied to the first visual layer and identify
configuration evidence relevant to the AdamW-versus-Muon comparison.

## First-convolution selection check

`muon_parameter_names` identifies the first `torch.nn.Conv2d` module and
excludes all of its direct parameters by identity before applying the normal
matrix-parameter predicate. The following runtime checks were observed:

| Model | First convolution parameter | Selected by Muon |
|---|---|---|
| ResNet-20 | `conv1.weight` | No |
| VGG-11 | `features.0.weight` | No |
| DenseNet-121 | `features.conv0.weight` | No |
| MobileNetV2 | `features.0.0.weight` | No |
| ViT-Tiny | `conv_proj.weight` | No |

The classifier/head parameters and one-dimensional normalization parameters
also remain in AdamW groups.

## Current optimization configuration

- AdamW and Muon both use base learning rate 0.0003.
- AdamW and Muon both use weight decay 0.01.
- Muon uses momentum 0.95, Nesterov updates, and five Newton--Schulz steps.
- Muon flattens each convolutional kernel to shape
  `out_channels x (in_channels * kernel_height * kernel_width)`.

For a matrix with dimensions `r x c`, the implementation applies the Muon
learning-rate scale below:

$$
\begin{equation}
\begin{aligned}
\eta_{\mathrm{Muon}}(r,c)
&= 0.0003 \times 0.2 \times \sqrt{\max(r,c)}
\end{aligned}
\end{equation}
$$

The resulting per-parameter ranges were:

| Model | Effective Muon learning-rate range | Matrices with smaller dimension at most 16 |
|---|---:|---:|
| ResNet-20 | 0.0003394–0.0014400 | 7 / 20 |
| VGG-11 | 0.0014400–0.0040729 | 0 / 7 |
| DenseNet-121 | 0.0006788–0.0020365 | 0 / 119 |
| MobileNetV2 | 0.0003394–0.0021466 | 19 / 51 |
| ViT-Tiny | 0.0008314–0.0016628 | 0 / 48 |

MobileNetV2 includes depthwise convolution matrices such as `32 x 9` and
`96 x 9`; their Muon orthogonalized updates are not representative of the
large dense matrices for which Muon was developed.

## Existing results

The claim that AdamW is higher on every task is not supported by the recorded
best-accuracy values. Muon has higher best accuracy on VGG-11 and ViT-Tiny;
AdamW is higher on ResNet-20, DenseNet-121, and MobileNetV2. AdamW has lower
final validation loss for all five current visual baselines.

## Diagnosis

The first-layer hypothesis is excluded by direct runtime enumeration. The
current result does not establish an optimizer ranking because it uses one
untuned base learning rate and one weight-decay value for both optimizers, and
it applies Muon to convolutional matrices of very different sizes. A valid
follow-up requires per-optimizer hyperparameter search and a parameter-group
ablation that sends depthwise or small convolution matrices to AdamW.
