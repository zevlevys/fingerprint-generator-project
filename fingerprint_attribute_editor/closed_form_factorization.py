import os
import argparse

import torch


def closed_form_factorization(args):
    # load checkpoint file
    ckpt = torch.load(args.checkpoint_path)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    # factorization
    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    # output result
    output_path = f"{args.exp_dir}/factor.pt"
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    torch.save({"ckpt": args.checkpoint_path, "eigvec": eigvec}, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract factor/eigenvectors of"
                                                 "latent spaces using closed form factorization")

    parser.add_argument("--exp_dir",
                        type=str,
                        default=".",
                        help="name of the result factor folder")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        help="path to checkpoint file")

    args = parser.parse_args()
    closed_form_factorization(args)
