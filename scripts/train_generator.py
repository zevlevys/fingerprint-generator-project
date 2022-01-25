import json
import math
import os
import pprint
import random

import torch
from torch import autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

# dataset
from configs import data_configs
from datasets.images_dataset import ImageDataset
from models.stylegan2.model import Generator, Discriminator
from options.generator_train_options import GeneratorTrainOptions
from training_utils.distributed import (
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from training_utils.non_leaking import augment, AdaptiveAugment


class GeneratorCoach:
    def __init__(self, opts):
        self.opts = opts
        self.device = "cuda"
        self.opts.latent = 512
        self.opts.n_mlp = 8
        self.opts.start_iter = 0

        # Initialize networks
        self.generator = Generator(self.opts.generator_image_size, self.opts.latent, self.opts.n_mlp,
                                   channel_multiplier=self.opts.channel_multiplier,
                                   is_gray=self.opts.is_gray).to(self.device)
        self.discriminator = Discriminator(self.opts.generator_image_size,
                                           channel_multiplier=self.opts.channel_multiplier,
                                           is_gray=self.opts.is_gray).to(self.device)
        self.g_ema = Generator(self.opts.generator_image_size, self.opts.latent, self.opts.n_mlp,
                               channel_multiplier=self.opts.channel_multiplier,
                               is_gray=self.opts.is_gray).to(self.device)
        self.g_ema.eval()

        accumulate(self.g_ema, self.generator, 0)

        if self.opts.checkpoint_path is not None:
            self.load_models()

        # Initialize optimizer
        g_reg_ratio = self.opts.g_reg_every / (self.opts.g_reg_every + 1)
        d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1)
        self.g_optim, self.d_optim = self.configure_optimizers(d_reg_ratio, g_reg_ratio)

        # Initialize dataset
        self.loader = self.configure_dataset()

        # Initialize outputs dir
        for output_folder in ['checkpoint', 'sample']:
            if not os.path.exists(f'{self.opts.exp_dir}/{output_folder}'):
                os.makedirs(f'{self.opts.exp_dir}/{output_folder}')

    def configure_dataset(self):
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        dataset = ImageDataset(target_root=dataset_args['train_target_root'],
                               target_transform=transforms_dict['transform_gt_train'],
                               opts=self.opts)
        loader = data.DataLoader(
            dataset,
            batch_size=self.opts.batch_size,
            sampler=data_sampler(dataset, shuffle=True),  # distributed=self.opts.distributed),
            drop_last=True,
        )
        return loader

    def load_models(self):
        print("load model:", self.opts.checkpoint_path)
        ckpt = torch.load(self.opts.checkpoint_path, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(self.opts.ckpt)
            self.opts.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        self.generator.load_state_dict(ckpt["g"])
        self.discriminator.load_state_dict(ckpt["d"])
        self.g_ema.load_state_dict(ckpt["g_ema"])
        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

    def configure_optimizers(self, d_reg_ratio, g_reg_ratio):
        g_optim = optim.Adam(
            self.generator.parameters(),
            lr=self.opts.learning_rate * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            self.discriminator.parameters(),
            lr=self.opts.learning_rate * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
        return g_optim, d_optim

    def train(self):

        loader = sample_data(self.loader)

        pbar = range(self.opts.max_steps)
        pbar = tqdm(pbar, initial=self.opts.start_iter, dynamic_ncols=True, smoothing=0.01)

        mean_path_length = 0

        r1_loss = torch.tensor(0.0, device=self.device)
        path_loss = torch.tensor(0.0, device=self.device)
        path_lengths = torch.tensor(0.0, device=self.device)
        self.mean_path_length_avg = 0
        loss_dict = {}

        self.g_module = self.generator
        self.d_module = self.discriminator

        accum = 0.5 ** (32 / (10 * 1000))
        self.ada_aug_p = self.opts.augment_p if self.opts.augment_p > 0 else 0.0

        if self.opts.augment and self.opts.augment_p == 0:
            ada_augment = AdaptiveAugment(self.opts.ada_target, self.opts.ada_length, 256, self.device)

        self.sample_z = torch.randn(self.opts.n_sample, self.opts.latent, device=self.device)

        for idx in pbar:
            i = idx + self.opts.start_iter

            if i > self.opts.max_steps:
                print("Done!")

                break

            real_img = next(loader)
            real_img = real_img.to(self.device)

            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = mixing_noise(self.opts.batch_size, self.opts.latent, self.opts.mixing, self.device)
            fake_img, _ = self.generator(noise)

            if self.opts.augment:
                real_img_aug, _ = augment(real_img, self.ada_aug_p, is_gray=self.opts.is_gray)
                fake_img, _ = augment(fake_img, self.ada_aug_p, is_gray=self.opts.is_gray)

            else:
                real_img_aug = real_img

            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            if self.opts.augment and self.opts.augment_p == 0:
                self.ada_aug_p = ada_augment.tune(real_pred)

            d_regularize = i % self.opts.d_reg_every == 0

            if d_regularize:
                real_img.requires_grad = True
                real_pred = self.discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)

                self.discriminator.zero_grad()
                (self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]).backward()

                self.d_optim.step()

            loss_dict["r1"] = r1_loss

            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            noise = mixing_noise(self.opts.batch_size, self.opts.latent, self.opts.mixing, self.device)
            fake_img, _ = self.generator(noise)

            if self.opts.augment:
                fake_img, _ = augment(fake_img, self.ada_aug_p, is_gray=self.opts.is_gray)

            fake_pred = self.discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

            loss_dict["g"] = g_loss

            self.generator.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            g_regularize = i % self.opts.g_reg_every == 0

            if g_regularize:
                path_batch_size = max(1, self.opts.batch_size // self.opts.path_batch_shrink)
                noise = mixing_noise(path_batch_size, self.opts.latent, self.opts.mixing, self.device)
                fake_img, latents = self.generator(noise, return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )

                self.generator.zero_grad()
                weighted_path_loss = self.opts.path_regularize * self.opts.g_reg_every * path_loss

                if self.opts.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                self.g_optim.step()

                self.mean_path_length_avg = (
                        reduce_sum(mean_path_length).item()
                )

            loss_dict["path"] = path_loss
            loss_dict["path_length"] = path_lengths.mean()

            accumulate(self.g_ema, self.g_module, accum)

            self.loss_reduced = reduce_loss_dict(loss_dict)

            self.validate(i, pbar)

            self.checkpoint_me(i)

    def checkpoint_me(self, i):
        if i % self.opts.save_interval == 0:
            torch.save(
                {
                    "g": self.g_module.state_dict(),
                    "d": self.d_module.state_dict(),
                    "g_ema": self.g_ema.state_dict(),
                    "g_optim": self.g_optim.state_dict(),
                    "d_optim": self.d_optim.state_dict(),
                    "args": self.opts,
                    "ada_aug_p": self.ada_aug_p,
                },
                f"{self.opts.exp_dir}/checkpoint/{str(i).zfill(6)}.pt",
            )

    def validate(self, i, pbar):
        d_loss_val = self.loss_reduced["d"].mean().item()
        g_loss_val = self.loss_reduced["g"].mean().item()
        r1_val = self.loss_reduced["r1"].mean().item()
        path_loss_val = self.loss_reduced["path"].mean().item()
        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                f"path: {path_loss_val:.4f}; mean path: {self.mean_path_length_avg:.4f}; "
                f"augment: {self.ada_aug_p:.4f}"
            )
        )
        if i % self.opts.image_interval == 0:
            with torch.no_grad():
                self.g_ema.eval()
                sample, _ = self.g_ema([self.sample_z])
                utils.save_image(
                    sample,
                    f"{self.opts.exp_dir}/sample/{str(i).zfill(6)}.png",
                    nrow=int(self.opts.n_sample ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )


def data_sampler(dataset, shuffle):

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


if __name__ == "__main__":
    # Set random seed
    random_seed = 1
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    opts = GeneratorTrainOptions().parse()

    if os.path.exists(opts.exp_dir):
        if len(os.listdir(opts.exp_dir)) > 1:
            ans = input('Oops... {} already exists. Do you wish to continue training_utils? [yes/no] '.format(opts.exp_dir))
            if ans == 'no':
                raise Exception('stop training_utils! Please change exp_dir argument.'.format(opts.exp_dir))
    else:
        os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    coach = GeneratorCoach(opts)
    coach.train()
