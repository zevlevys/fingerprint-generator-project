dataset_paths = {
    # NIST SD14 Reconstruction
    'nist_sd14_mnt_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/train_A',
    'nist_sd14_mnt_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/test_A',
    'nist_sd14_mnt_gt_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/train_B',
    'nist_sd14_mnt_gt_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/test_B',

    # NIST SD4 Reconstruction
    'nist_sd4_mnt_train': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/train_A',
    'nist_sd4_mnt_test': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/test_A',
    'nist_sd4_mnt_gt_train': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/train_B',
    'nist_sd4_mnt_gt_test': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/test_B',

    # NIST SD14 Synthesis
    'nist_sd14_gt_train': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/train_B',
    'nist_sd14_gt_test': '/hdd/PycharmProjects/fingerprints/Data/CROPPED_NIST14_with_ori/test_B',

    # NIST SD4 Synthesis
    'nist_sd4_gt_train': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/train_B',
    'nist_sd4_gt_test': '/hdd/PycharmProjects/fingerprints/Data/NISTSD4/NISTSD4_for_PIX2PIX/test_B',
}

model_paths = {
    'stylegan_weights': '/content/drive/MyDrive/OAI/style_gan_weights.pt',
    'encoder_pretrained': '/content/drive/MyDrive/OAI/model_ir_se50.pth',
    'fingernet': '/content/drive/MyDrive/OAI/new_model_format.pth'
}
