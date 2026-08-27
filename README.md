# Ideas

- [x] Adaptive Dropout Rate for each layer

- Alternative Update: Adam -> GN -> Adam (With momentum saved) -> GN -> ...

## Compare Adam/Muon with GN in important layers

- Judge whether Adam is noisy

$$
\begin{equation}
\begin{aligned}
R_{noise} = \frac{\left( \|\Delta_W^a - \Delta_W^b \| \right)^2}{ 2 \left( \frac{\|\Delta_W^a + \Delta_W^b\|}{2} \right)^2 + \epsilon}
\end{aligned}
\end{equation}
$$

# Workflow

## Adam warm-up for randomly initialized models

Run AdamW until the model reaches a reasonably stable region and gradient/activation statistics settle.
This does not initialize the weights; GN performs a fresh local model linearization during every GN outer step.
For pretrained models, skip this phase or use a short calibration warm-up if the task head is new.

## Layer-wise GN bootstrap

Run several accepted layer-wise GN outer updates.
GN computes a curvature-aware direction independently for each layer, merges those directions, and performs global line search.
This establishes a reliable initial global update direction, not a stored “global gradient.”

## GN-to-Adam bridge

Store the accepted GN direction per layer, GN warm-start state, and before/after parameter-gradient pairs.
Use the normalized GN direction only as a short, decaying directional prior.
Do not insert it into Adam’s momentum buffers. Adam moments must be built from real gradients.

## Fast Adam phase with low-cost monitoring

Adam performs the normal training updates.
Periodically sample a rotating, representative set of layers—early, middle, late, embedding, and output head.
Combine sampled-layer noise with global loss progress and Adam/reference-gradient alignment.
A sampled-layer warning alone must not trigger GN; confirm it with a larger reference-batch GN trial.

## Recurrent GN correction

Enter GN only when Adam is persistently noisy or poorly aligned, progress has stalled, and a GN trial produces a valid improvement.
Run a bounded number of accepted GN corrections while continuing to update shadow Adam state from real gradients.
Return through the GN-to-Adam bridge when shadow Adam is aligned again and its improvement per second approaches GN’s.

## Final-mile GN recovery

At the nominal end of training, check validation accuracy against:
the configured target accuracy; and
the configured minimum accuracy improvement over recent evaluations.
If accuracy already meets the target, stop—even if its delta is small.
If accuracy remains below target and has plateaued, restore the best checkpoint and run conservative GN recovery with stronger damping and smaller steps.
Keep only validation-improving checkpoints and roll back if GN fails to improve quality.
