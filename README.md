# 🏞️ Seraena

Seraena is 🚧 WIP 🚧 PyTorch code for stably training mode-dropping deterministic autoencoders like TAESD using only conditional GAN training in the decoder (no LPIPS/L1).

<a href="TAESDXL_Training_Example.ipynb">Sample TAESDXL decoder training code using Seraena</a>

This code is still pretty basic (and I haven't benchmarked the resulting models against the TAESDXL 1.2 checkpoints), but after training a few hours on a small dataset it seemed to work well enough.

![](./screenshot.png)
