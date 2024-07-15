# Interpretable Latent Spaces in GANs Using Space-Filling Vector Quantization

This repository contains a demo to test and compare interpretable directions found by our proposed method, GANSpace, and LatentCLR methods in intermediate latent space (W) of StyleGAN2 pretrained on FFHQ dataset. The paper is submitted to ECCV 2024.

# requirements
Plase install the requirements using the following lines in your terminal window:

`conda create --name eccv2024_sfvq python=3.9`

`conda activate eccv2024_sfvq`

`pip install -r requirments.txt`

Also, please download the StyleGAN2 pretrained model named "stylegan2-ffhq-1024x1024.pkl" from NVIDIA website under [this link.](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2/files)

In addition, please extract the files existing in 'files.zip'. Please keep the pretrained model and extracted files in the same directory.

# Demo

In 'demo.py' code, you only need to change (play with) "direction_name" and "sigma_list" variables to test interpretable directions over different shift values ($\sigma$).
