import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image
from tqdm.auto import tqdm
from ddpm import NoiseScheduler
from model import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

def gen(model, sample_shape, config, capture_gap = 50):
    frames = []
    model.to(device)
    model.eval()
    sample = torch.randn(sample_shape).unsqueeze(0)
    sample = sample.to(device)
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule
    )
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.from_numpy(np.repeat(t, 1)).long()
        t = t.to(device)
        num = torch.from_numpy(np.repeat(7, 1)).long()
        num = num.to(device)
        with torch.no_grad():
            residual = model(sample, t, num)
        sample = noise_scheduler.step(residual.reshape(sample_shape), t, sample)
        if i%capture_gap == 0 or i == config.num_timesteps-1:
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
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    config = parser.parse_args()

    model = UNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    path = "exps/base/model.pth"
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    gen(model=model, sample_shape=(28, 28), config=config)