import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im
from torchvision.utils import make_grid
from options.inference_mnt_encoder_options import MntEncoderInferenceOptions
from models.fingergen import FingerGen


def run():
    test_opts = MntEncoderInferenceOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    out_path_inputs = os.path.join(test_opts.exp_dir, 'inference_inputs')

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    opts = Namespace(**opts)

    if opts.couple_outputs:
        os.makedirs(out_path_coupled, exist_ok=True)
    else:
        os.makedirs(out_path_results, exist_ok=True)
        os.makedirs(out_path_inputs, exist_ok=True)

    net = FingerGen(opts, opts.resize_factor)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    images = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)
        
        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            source = Image.open(im_path)
            resize_amount = (int(opts.resize_factor), int(opts.resize_factor)) if opts.resize_outputs else (512, 512)

            if opts.couple_outputs:
                
                res = np.concatenate([np.array(source.resize(resize_amount)),
                                      np.array(result.resize(resize_amount))], axis=1)
                if opts.output_mode == "grid":
                    images.append(res)
                else:
                    Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            else:
                # save result
                output_im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
                if opts.resize_factor is not None:
                    result = result.resize(resize_amount)
                Image.fromarray(np.array(result)).save(output_im_save_path, dpi=(500, 500))

                # save input mnt
                input_im_save_path = os.path.join(out_path_inputs, os.path.basename(im_path))
                if opts.resize_factor is not None:
                    source = source.resize(resize_amount)
                Image.fromarray(np.array(source)).save(input_im_save_path, dpi=(500, 500))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    if opts.output_mode == "grid":
        grid = make_grid(images, nrow=3)
        grid_im = tensor2im(grid)
        Image.fromarray(grid_im).save(os.path.join(out_path_coupled, "grid.png"))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
