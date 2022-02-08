from argparse import ArgumentParser


class AttributeEditorOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        ''' arguments for attribute_editor script '''

        self.parser.add_argument("--index",
                                 type=int,
                                 default=0,
                                 help="index of eigenvector")
        self.parser.add_argument("--degree",
                                 type=float,
                                 default=5,
                                 help="scalar factors for moving latent vectors along eigenvector",
                                 )
        self.parser.add_argument("--channel_multiplier",
                                 type=int,
                                 default=2,
                                 help='channel multiplier factor. config-f = 2, else = 1',
                                 )
        self.parser.add_argument("--checkpoint_path",
                                 type=str,
                                 required=True,
                                 help="stylegan2 checkpoints")
        self.parser.add_argument("--size",
                                 type=int,
                                 default=256,
                                 help="output image size of the generator"
                                 )
        self.parser.add_argument("--n_sample",
                                 type=int,
                                 default=7,
                                 help="number of samples created"
                                 )
        self.parser.add_argument("-num_out", "--number_of_outputs",
                                 type=int,
                                 default=1,
                                 help="number of samples created"
                                 )
        self.parser.add_argument("--truncation",
                                 type=float,
                                 default=0.7,
                                 help="truncation factor"
                                 )
        self.parser.add_argument("--device",
                                 type=str,
                                 default="cuda",
                                 help="device to run the model"
                                 )
        self.parser.add_argument("--out_prefix",
                                 type=str,
                                 default="fingerprint",
                                 help="filename prefix to result samples",
                                 )
        self.parser.add_argument("--exp_dir",
                                 type=str,
                                 default="./",
                                 help="filename prefix to result samples",
                                 )
        self.parser.add_argument("--factor_path",
                                 type=str,
                                 help="name of the closed form factorization result factor file",
                                 )
        self.parser.add_argument("--is_gray",
                                 action="store_true",
                                 help="generate gray images",
                                 )
        self.parser.add_argument("--resize_factor",
                                 type=int,
                                 help="output size",
                                 )
        self.parser.add_argument("--concat_output_images",
                                 action="store_true",
                                 help="concatenate output images together",
                                 )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
