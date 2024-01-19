## Repository  Details

This repository hosts the code for a Norwegian University of Science and Technology project thesis:
[Barlow-Twins-VQVAE.pdf](https://github.com/erlendlokna/Barlow-Twins-VQVAE/files/13859816/Barlow-Twins-VQVAE.pdf)


It builds on the work of the [TimeVQVAE implementation](https://github.com/ML4ITS/TimeVQVAE) and further explores the representations of Vector Quantized Variational Autoencoders (VQVAEs) in time series analysis.

In this project, we've integrated Self-Supervised Learning (SSL) techniques, drawing inspiration from both the foundational work of TimeVQVAE and insights from the [Barlow Twins paper](https://arxiv.org/abs/2103.03230). Our focus lies in enhancing VQVAE models with SSL to improve their performance and efficiency. The Barlow Twins implementation proved to enhance the VQVAEs robustness and ability to generalize. 

## Contents
This repository is structured around two main models:
1. **VQVAE with STFT**: A traditional VQVAE model that utilizes Short-Time Fourer Transform (STFT) for processing time-frequency representations
2. **Barlow Twins modified VQVAE**: A VQVAE model featuring a dual-branched encoder. This model is augmented with a Barlow Twins SSL objective, acting as a regularizer to boost training effectiveness and yield more distinct latent representations.

## Model Illustrations
Both models operate on the principle of compressing time series input $x$ into latent variables $z$ using time-frequency representations $u$. These models aim to reconstruct the input $\hat{x}$ through a codebook $\mathcal{Z}$ and an $\textit{argmin process}$. 

**VQVAE using STFT**:

![VQVAE](https://github.com/erlendlokna/Barlow-Twins-VQVAE/assets/80318998/d29e1f57-114d-4e62-b29b-c7cdab69942c)

**Two-Branched barlow Twins VQVAE**:

![BarlowTwinsVQVAE](https://github.com/erlendlokna/Barlow-Twins-VQVAE/assets/80318998/574e021b-6eab-45e4-bd75-0b61c956ee14)

## Running The Code
### Experiment for Thesis
> python download_ucr_dataset.py 

Downloads the UCR archive.

> python experiment.py 

Executes the thesis experiment. Training and  comparing the VQVAE with the two-branced Barlow Twins VQVAE for different SSL loss weightings, $\gamma$.

### Training Individual Models
> python train_VQVAE.py 

Initiates the training for the traditional VQVAE model.

> python train_BTVQVAE.py

Begins the training process for the Barlow Twins-augmented VQVAE

## Contrubution
Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
