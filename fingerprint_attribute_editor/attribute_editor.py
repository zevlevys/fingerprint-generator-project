import os
import random
from tqdm import tqdm

import torch
from torchvision import utils

from fingerprint_attribute_editor.attribute_editor_options import AttributeEditorOptions
from models.stylegan2.model import Generator


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    import math
    irange = range

    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding) \
                .narrow(2, x * width + padding, width - padding) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp - A filename(string) or file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    if args.resize_factor is not None:
        im = im.resize((int(args.resize_factor), int(args.resize_factor)))
    im.save(fp, format=format)


def attribute_editor(args):
    print('load model and params...')
    eigvec = torch.load(args.factor_path)["eigvec"].to(args.device)
    ckpt = torch.load(args.checkpoint_path)

    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier, is_gray=args.is_gray
                  ).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    # create output folder, if needed
    output_path = f'{args.exp_dir}/index_{args.index}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f'generating {args.number_of_outputs} fingerprint images...\n output folder: {output_path}')
    for sample_index in tqdm(range(args.number_of_outputs)):
        latent = torch.randn(args.n_sample, 512, device=args.device)
        latent = g.get_latent(latent)

        direction = args.degree * eigvec[:, args.index].unsqueeze(0)

        # original
        img, _ = g(
            [latent],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        # forward
        img1, _ = g(
            [latent + direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        # backward
        img2, _ = g(
            [latent - direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )

        # output results
        if args.concat_output_images:
            utils.save_image(torch.cat([img1, img, img2], 0),
                             f"{output_path}/{args.out_prefix}_sample_-{sample_index}_degree-{args.degree}.png",
                             normalize=True,
                             range=(-1, 1),
                             nrow=args.n_sample)

        else:
            for image, image_name in zip([img, img1, img2], ['original', 'backward', 'forward']):
                save_image(image,
                           f"{output_path}/{args.out_prefix}_{sample_index}_{image_name}.jpg",
                           normalize=True,
                           range=(-1, 1),
                           nrow=args.n_sample)


if __name__ == "__main__":
    # Set random seed
    random_seed = 1
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    torch.set_grad_enabled(False)

    args = AttributeEditorOptions().parse()
    attribute_editor(args)
