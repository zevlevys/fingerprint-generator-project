from abc import abstractmethod

import torchvision.transforms as transforms


class TransformsConfig(object):
    def __init__(self, opts):
        self.opts = opts

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
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target)]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize(self.normalization_vec_source, self.normalization_vec_source)
            ]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(self.normalization_vec_target, self.normalization_vec_target)]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                # transforms.Normalize(self.normalization_vec_source, self.normalization_vec_source)
            ])
        }
        return transforms_dict
