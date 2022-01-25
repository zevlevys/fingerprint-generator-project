from abc import abstractmethod

import torchvision.transforms as transforms


class TransformsConfig(object):
    def __init__(self, opts):
        self.opts = opts

        if hasattr(self.opts, 'input_nc'):
            if self.opts.input_nc == 1:
                self.normalization_vec_source = 0.5
            else:
                self.normalization_vec_source = [0.5, 0.5, 0.5]

        if self.opts.label_nc == 1:
            self.normalization_vec_target = 0.5
        else:
            self.normalization_vec_target = [0.5, 0.5, 0.5]

    @abstractmethod
    def get_transforms(self):
        pass


class MntToFingerTransforms(TransformsConfig):
    def __init__(self, opts):
        super(MntToFingerTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),  # TODO: replace with args.size
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target)]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # No normalization for minutiae maps
            ]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target)]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # No normalization for minutiae maps
            ])
        }
        return transforms_dict


class FingerprintSynthesisTransforms(TransformsConfig):
    def __init__(self, opts):
        super(FingerprintSynthesisTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target, inplace=True)
                ]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 256)),  # TODO: replace with args.size
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target, inplace=True)
                ]),
            'transform_inference': None
        }
        return transforms_dict
