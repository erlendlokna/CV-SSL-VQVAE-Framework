## Repository details
This repository contains the code for a Norwegian University of Science and Technology project thesis.

Implementation is heavily inspired by the [TimeVQVAE implementation](https://github.com/ML4ITS/TimeVQVAE). Here they introduce the TimeVQVAE model incorporating the STFT and ISTFT.

This repository focuses on the addition of SSL, using the groundwork done by the authors of TimeVQVAE and the paper [Vector Quantized Time Series Generation with a Bidirectional Prior Model](https://arxiv.org/abs/2303.04743).

## Contents
This repository contains two models:
1. Vector Quantized Variational Autoencoder (VQVAE) using STFT to process time-frequency representations.
2. A VQVAE model using a two-branched encoder, enabling SSL loss functions to alter its representation capabilities. In this implementation, we introduce a Barlow Twins SSL objective for the VQVAE encoder, acting as a regularizer for enhanced training efficiency and more salient latent representations.

## Model illustrations:
Both models use time-frequency representations ($u$) to compress the input $x$ (time series) into compact latent variables $z$. They are optimized to reconstruct the input $\hat{x}$, using a codebook $\mathcal{Z}$ and the argmin process.

**VQVAE using STFT**:

![VQVAE](https://github.com/erlendlokna/Barlow-Twins-VQVAE/assets/80318998/d29e1f57-114d-4e62-b29b-c7cdab69942c)

**Barlow Twins modified VQVAE**:

![BarlowTwinsVQVAE](https://github.com/erlendlokna/Barlow-Twins-VQVAE/assets/80318998/574e021b-6eab-45e4-bd75-0b61c956ee14)


## Running code

### Experiment for Thesis
> experiment.py

Runs the experiment on the UCR archive. It requires a data folder containing the UCR archive. Alter src/configs/config.yaml to change the training setup and procedure.

### Training models
> train_barlowvqvae.py

Starts the training procedure for the dual-branched VQVAE using the Barlow Twins objective.

> train_vqvae.py

Starts the training procedure for the VQVAE.
