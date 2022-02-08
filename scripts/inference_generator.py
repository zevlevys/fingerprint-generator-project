import os
import random
import numpy as np

import torch
from utils.common import tensor2im
from tqdm import tqdm
from PIL import Image

from models.stylegan2.model import Generator
from options.inference_generator_options import GeneratorInferenceOptions


def run():
    test_opts = GeneratorInferenceOptions().parse()

    if not os.path.exists(test_opts.exp_dir):
        os.makedirs(test_opts.exp_dir)

    test_opts.latent = 512
    test_opts.n_mlp = 8

    g_ema = Generator(
        test_opts.generator_image_size, test_opts.latent, test_opts.n_mlp,
        channel_multiplier=test_opts.channel_multiplier, is_gray=test_opts.is_gray
    ).to(device)
    checkpoint = torch.load(test_opts.checkpoint_path)

    g_ema.load_state_dict(checkpoint["g_ema"])

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(test_opts.n_images)):
            sample_z = torch.randn(test_opts.sample, test_opts.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=1.0, truncation_latent=None
            )

            result = tensor2im(sample[0])

            if test_opts.resize_factor is not None:
                result = result.resize((int(test_opts.resize_factor), int(test_opts.resize_factor)))
            im_save_path = f"{test_opts.exp_dir}/{str(i).zfill(6)}.png"
            Image.fromarray(np.array(result)).save(im_save_path, dpi=(500, 500))


if __name__ == "__main__":
    # Set random seed
    random_seed = 1
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    device = "cuda"

    run()
