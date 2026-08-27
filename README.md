# Ideas

- Adaptive Dropout Rate for each layer

- GN -> Adam (With momentum saved)

- Alternative Update: GN -> Adam (With momentum saved) -> GN -> ...

- Compare with GN in important layers



# Workflow

1. For a non-pretrained model, run several rounds of Adam to linearlize its weight, else skip this step
2. Then run several rounds of GN optimizer, build up the initial global gradient
3. Store the GN's normalized gradient, start switching to Adam with a momentum derived from the stage of GN
4. Run fast SGD with Adam, while sample a few layer's gradient and compare with the original GN's gradient. No full comparation for save the compute cost.
5. When the gradient difference is too large, switch back to GN for several rounds to correct the gradient direction
6. In final stage when the delta accuracy is too small, switch to GN to improve the final quality.