from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'nist_sd14_mnt': {
		'transforms': transforms_config.MntToFingerTransforms,
		'train_source_root': dataset_paths['nist_sd14_mnt_train'],
		'train_target_root': dataset_paths['nist_sd14_mnt_gt_train'],
		'test_source_root': dataset_paths['nist_sd14_mnt_test'],
		'test_target_root': dataset_paths['nist_sd14_mnt_gt_test'],
	},
	'nist_sd4_mnt': {
		'transforms': transforms_config.MntToFingerTransforms,
		'train_source_root': dataset_paths['nist_sd4_mnt_train'],
		'train_target_root': dataset_paths['nist_sd4_mnt_gt_train'],
		'test_source_root': dataset_paths['nist_sd4_mnt_test'],
		'test_target_root': dataset_paths['nist_sd4_mnt_gt_test'],
	},
	'nist_sd14_synthesis': {
		'transforms': transforms_config.FingerprintSynthesisTransforms,
		'train_target_root': dataset_paths['nist_sd14_gt_train'],
		'test_target_root': dataset_paths['nist_sd14_gt_test'],
	},
	'nist_sd4_synthesis': {
		'transforms': transforms_config.FingerprintSynthesisTransforms,
		'train_source_root': dataset_paths['nist_sd4_mnt_train'],
		'train_target_root': dataset_paths['nist_sd4_mnt_gt_train'],
		'test_source_root': dataset_paths['nist_sd4_mnt_test'],
		'test_target_root': dataset_paths['nist_sd4_mnt_gt_test'],
	},
  'retaining_id': {
    		'transforms': transforms_config.MntToFingerTransforms,
    		'train_source_root': '/content/train_cropped_maps',
		'train_target_root': '/content/content/retaining_id/train_cropped',
		'test_source_root': '/content/test_cropped_maps',
		'test_target_root': '/content/content/retaining_id/test_cropped',
  }
}
