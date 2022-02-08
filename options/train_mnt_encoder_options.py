from argparse import ArgumentParser

from configs.paths_config import model_paths


class MntEncoderTrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # Paths and Data params
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='nist_sd14_mnt', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to pretrained FingerGen model checkpoint')
        self.parser.add_argument('--stylegan_weights', default=model_paths['stylegan_weights'], type=str,
                                 help='Path to StyleGAN model weights. used only when checkpoint_path is None')

        # Architecture params
        self.parser.add_argument('--generator_image_size', default=256, type=int,
                                 help='Image size for stylegan2 generator')
        self.parser.add_argument('--style_count', default=14, type=int, help='Number of style vectors')
        self.parser.add_argument('--is_gray', default=False, type=bool, help='generate greyscale output')
        self.parser.add_argument('--input_nc', default=3, type=int,
                                 help='Number of input image channels to the encoder')
        self.parser.add_argument('--label_nc', default=1, type=int,
                                 help='Number of input label channels to the encoder')

        # Optimizer
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space insteaf of w+')

        # Losses
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--fingernet_lambda', default=0, type=float, help='fingernet loss multiplier factor')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')

        # Logging
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

        # Batch and Workers
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
