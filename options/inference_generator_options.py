from argparse import ArgumentParser


class GeneratorInferenceOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        ''' arguments for inference_generator script '''

        # Paths
        self.parser.add_argument('--exp_dir',
                                 type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path',
                                 default=None,
                                 type=str,
                                 help='Path to FingerGen model checkpoint')

        # Architecture params
        self.parser.add_argument('--generator_image_size',
                                 default=256,
                                 type=int,
                                 help='Image size for stylegan2 generator')
        self.parser.add_argument("--channel_multiplier",
                                 type=int,
                                 default=2,
                                 help="channel multiplier of the generator. config-f = 2, else = 1",
                                 )
        self.parser.add_argument('--is_gray',
                                 action="store_true",
                                 help='generate greyscale output')

        # Output params
        self.parser.add_argument('--resize_factor',
                                 default=512,
                                 type=int,
                                 help='outputs resize to (resize_factor X resize_factor)')

        # Number of images to generate
        self.parser.add_argument('--n_images',
                                 type=int,
                                 default=None,
                                 help='Number of images to output. If None, run on all data')
        self.parser.add_argument("--sample",
                                 type=int,
                                 default=1,
                                 help="number of samples to be generated for each image",
                                 )

        sef.parser_add_argument('--output_mode',
                                type=string,
                                default='save',
                                help="Save the images from inference or plot show them"
                               )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
