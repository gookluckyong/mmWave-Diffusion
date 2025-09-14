# mmWave-Diffusion
Diffusion Framework for 1D Signals with Radar Diffusion Transforme (ICASSP 2026 Submission)
Note
This repository contains the core diffusion framework and the Radar Diffusion Transformer (RDT) model implementation from our ICASSP 2026 submission.
The paper is currently under review.
This release is intended as a research preview, for academic exchange and reproducibility.
Dataset loaders and experiment scripts are not included.

✨ Features

Observation-Anchored Diffusion Process

Forward process injects residuals between clean and observed signals.

Reverse process with closed-form posterior.

Exponential η-schedule with power adjustment.

Supports weighted MSE loss.

Spaced Diffusion (Fast Sampling)

Train with 1000 steps, sample with e.g. 20 steps.

Automatic remapping between reduced and original timesteps.

Radar Diffusion Transformer (RDT)

Patch-based embedding of 1D signals.

Self-attention + cross-attention with learnable gate.

Local banded mask for cross-attention.

AdaLN-Zero modulation for stable training.

📂 Repository Structure

basic_ops.py         # Utility layers: SiLU, GroupNorm32, timestep embeddings

gaussian_diffusion.py# Diffusion forward / backward / training / sampling

respace.py           # SpacedDiffusion: step-skipping wrapper for fast sampling

script_util.py       # Helper to create diffusion instances

network.py           # Radar Diffusion Transformer (RDT)

🚀 Getting Started
Installation

Requires Python 3.9+ and PyTorch ≥ 1.12:

cd RadarDiffusion-RDT

pip install torch numpy

Minimal Example

import torch

from network import RDT

from script_util import create_gaussian_diffusion

# Diffusion process
diffusion = create_gaussian_diffusion(
    schedule_name='exponential',
    steps=1000,
    timestep_respacing=50,
    kappa=0.7,
    min_noise_level=1e-3,
    schedule_kwargs={'power': 0.3},
    loss_type='mse'
)

# RDT model
model = RDT(
    sequence_length=400,
    in_channels=1,
    cond_channels=1,
    out_channels=1,
    patch_size=20,
    hidden_size=256,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0,
    cross_local_window=1
)

# Dummy data
x0 = torch.randn(8, 1, 400)   # clean target
y  = torch.randn(8, 1, 400)   # conditional input
t  = torch.randint(0, diffusion.num_timesteps, (8,))

# Training loss
terms, x_t, pred_x0 = diffusion.training_losses(model, x0, y, t)
loss = terms['mse'].mean()
loss.backward()

# Sampling
with torch.no_grad():
    recon = diffusion.p_sample_loop(y, model, progress=True)
print(recon.shape)  # [8, 1, 400]

⚙️ Key Hyperparameters

Diffusion

steps: training steps (e.g. 1000)

timestep_respacing: sampling steps (e.g. 20)

schedule_name: noise schedule (exponential)

kappa: variance scaling

loss_type: mse or weighted_mse

RDT

sequence_length: input length (must be divisible by patch_size)

patch_size: patch size for Conv1d embedding

hidden_size: Transformer embedding dimension

depth: number of Transformer blocks

num_heads: attention heads

cross_local_window: width of local cross-attention mask

📜 License

This project is released under the MIT License. See LICENSE
 for details.

🔜 Future Work / TODO

 Release dataset preprocessing pipeline

 Add full training & evaluation scripts

 Provide pretrained model checkpoints

 Update citation once the ICASSP 2026 review is completed
