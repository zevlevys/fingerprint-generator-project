from argparse import ArgumentParser


class MntEncoderInferenceOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        ''' arguments for inference_mnt_encoder script '''

        # Paths
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to FingerGen model checkpoint')
        self.parser.add_argument('--data_path', type=str, default='gt_images',
                                 help='Path to directory of images to evaluate')

        # input params
        self.parser.add_argument('--input_nc', default=3, type=int, help='number of channels for input images')

        # output params
        self.parser.add_argument('--couple_outputs', action='store_true',
                                 help='Whether to also save inputs + outputs side-by-side')
        self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs')
        self.parser.add_argument('--resize_factor', default=512, type=int,
                                 help='outputs resize to (resize_factor X resize_factor)')

        # Number of images and Batch size
        self.parser.add_argument('--n_images', type=int, default=None,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        # Style-Mixing
        self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
        self.parser.add_argument('--latent_mask', type=str, default=None,
                                 help='Comma-separated list of latents to perform style-mixing with')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
