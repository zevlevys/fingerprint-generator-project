from argparse import ArgumentParser


class GeneratorTrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # Paths and Data params
        self.parser.add_argument('--exp_dir',
                                 type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type',
                                 default='nist_sd14_synthesis',
                                 type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--checkpoint_path',
                                 default=None,
                                 type=str,
                                 help='Path to pretrained StyleGAN model checkpoint')

        # Architecture params
        self.parser.add_argument('--generator_image_size',
                                 default=256,
                                 type=int,
                                 help='Image size for stylegan2 generator')
        self.parser.add_argument('--is_gray',
                                 action="store_true",
                                 help='generate greyscale output')
        self.parser.add_argument('--label_nc',
                                 default=1,
                                 type=int,
                                 help='Number of label channels')
        self.parser.add_argument("--channel_multiplier",
                                 type=int,
                                 default=2,
                                 help="channel multiplier factor for the model. config-f = 2, else = 1")

        # Losses
        self.parser.add_argument("--path_regularize",
                                 type=float,
                                 default=2,
                                 help="weight of the path length regularization")
        self.parser.add_argument('--path_batch_shrink',
                                 type=int,
                                 default=2,
                                 help='batch size reducing factor for the path length regularization'
                                      '(reduce memory consumption)')
        self.parser.add_argument("--r1",
                                 type=float,
                                 default=10,
                                 help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every",
                                 type=int,
                                 default=16,
                                 help="interval of the applying r1 regularization")
        self.parser.add_argument("--g_reg_every",
                                 type=int,
                                 default=4,
                                 help="interval of the applying path length regularization",
                                 )
        self.parser.add_argument("--mixing",
                                 type=float,
                                 default=0.9,
                                 help="probability of latent code mixing")

        # Augmentations
        self.parser.add_argument("--augment",
                                 action="store_true",
                                 help="apply non leaking augmentation"
                                 )
        self.parser.add_argument("--augment_p",
                                 type=float,
                                 default=0,
                                 help="probability of applying augmentation. 0 = use adaptive augmentation",
                                 )
        self.parser.add_argument("--ada_target",
                                 type=float,
                                 default=0.6,
                                 help="target augmentation probability for adaptive augmentation",
                                 )
        self.parser.add_argument("--ada_length",
                                 type=int,
                                 default=500 * 1000,
                                 help="target during to reach augmentation probability for adaptive augmentation",
                                 )
        self.parser.add_argument("--ada_every",
                                 type=int,
                                 default=256,
                                 help="probability update interval of the adaptive augmentation",
                                 )

        # Logging
        self.parser.add_argument('--max_steps',
                                 default=500000,
                                 type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--image_interval',
                                 default=100,
                                 type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--save_interval',
                                 default=10000,
                                 type=int,
                                 help='Model checkpoint interval')
        self.parser.add_argument("--n_sample",
                                 type=int,
                                 default=64,
                                 help="number of the samples generated during training",
                                 )

        # Batch size and lr
        self.parser.add_argument('--batch_size',
                                 default=4,
                                 type=int,
                                 help='Batch size for training')

        self.parser.add_argument('--learning_rate',
                                 default=0.002,
                                 type=float,
                                 help='Optimizer learning rate')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
