# Seraena Model Checkpoints

This folder contains some trained model checkpoints (autoencoder + discriminator) made using Seraena.

## Checkpoint folders

* [`./taesdxl_photos_96`](./taesdxl_photos_96) contains checkpoints from the basic taesdxl training notebook trained from scratch for ~300k steps (on photos cropped to 96px resolution).
* [`./taesdxl_photos_256`](./taesdxl_photos_256) contains checkpoints from the basic taesdxl training notebook finetuned for ~10k steps on photos cropped to 256px resolution (starting from the checkpoint from `taesdxl_photos_96`).

## Files in each checkpoint folder

* `model.pth` contains the autoencoder weights
* `seraena.pth` contains the discriminator weights
