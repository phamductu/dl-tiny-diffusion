import argparse
import os

import torch
from tqdm.auto import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from positional_embeddings import PositionalEmbedding
import datasets

from ddpm import NoiseScheduler, CNN
from torchvision.utils import save_image

def gen(model, sample_shape, config):
    frames = []
    model.eval()
    sample = torch.randn(sample_shape).unsqueeze(0)
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule
    )
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.from_numpy(np.repeat(t, 1)).long()
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual.reshape(sample_shape), t, sample)
        frames.append(sample.squeeze(0))

    print("Saving images...")
    outdir = f"exps/{config.experiment_name}"
    imgdir = f"{outdir}/reverse-process"
    os.makedirs(imgdir, exist_ok=True)
    for i, frame in enumerate(frames):
        save_image(frame, f"{imgdir}/i{i:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["circle", "dino", "line", "moons", "mnist"])
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    config = parser.parse_args()

    model = CNN()

    path = "exps/base/model.pth"
    model.load_state_dict(torch.load(path))
    model.eval()
    gen(model=model, sample_shape=(28, 28), config=config)