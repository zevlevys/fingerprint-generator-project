# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse, spatial

import matplotlib
matplotlib.use('Qt5Agg')

from configs.paths_config import model_paths


class FingerNetLoss(nn.Module):
    def __init__(self, label_nc=3):
        super(FingerNetLoss, self).__init__()

        self.finger_net = FingerNet()
        self.finger_net.load_state_dict(
            torch.load(model_paths['fingernet']))
        self.finger_net.cuda()

        self.criterion = nn.L1Loss()

        # weights for outputs: ori_out_2, upsample_ori, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out
        self.weights = [1.0/2, 1.0/2, 1.0, 1.0, 1.0, 1.0]

        self.th = 0.5
        self.mask_mult = 1
        self.mask_bias = 0.1
        self.label_nc = label_nc

    def forward(self, x, y):
        re = torch.nn.UpsamplingBilinear2d(512)
        y = re(y)
        x = re(x)
        x_fnet, y_fnet = self.finger_net(x, label_nc=self.label_nc), self.finger_net(y, label_nc=self.label_nc)

        y_mask = (y_fnet[5] > self.th).type_as(y_fnet[5])
        y_mask = self.mask_bias + self.mask_mult*y_mask

        loss = 0
        for i in range(len(x_fnet)):
            if i == 1:
                continue
            loss += self.weights[i] * self.criterion(x_fnet[i]*y_mask, y_fnet[i].detach()*y_mask)

        loss += self.criterion(x*re(y_mask), y.detach()*re(y_mask))

        return loss

    @staticmethod
    def sum_abs_diff(a, b):
        return abs(a - b).sum()

    @staticmethod
    def l1_with_mask(a, b, x_mask, y_mask):
        return abs(a * x_mask - b * y_mask).sum() / (y_mask.sum() + 1)

    def label_2_mnt(self, output_fnet):
        output_fnet_np = []
        for i in range(len(output_fnet)):
            output_fnet_np.append(output_fnet[i].detach().cpu().numpy())
        mnt = self.label2mnt(output_fnet_np[5], output_fnet_np[3], output_fnet_np[4], output_fnet_np[2])
        mnt_nms = self.nms(mnt)
        return mnt_nms

    # currently can only produce one each time
    def label2mnt(self, mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
        mnt_s_out = np.squeeze(mnt_s_out)
        mnt_w_out = np.squeeze(mnt_w_out)
        mnt_h_out = np.squeeze(mnt_h_out)
        mnt_o_out = np.squeeze(mnt_o_out)
        assert len(mnt_s_out.shape) == 2 and len(mnt_w_out.shape) == 3 and len(mnt_h_out.shape) == 3 and len(
            mnt_o_out.shape) == 3
        # get cls results
        mnt_sparse = sparse.coo_matrix(mnt_s_out > thresh)
        mnt_list = [(r, c) for r, c in zip(mnt_sparse.row, mnt_sparse.col)]
        mnt_list = np.array(mnt_list, dtype=np.int32)
        if mnt_list.shape[0] == 0:
            return np.zeros((0, 4))
        # get regression results
        mnt_w_out = np.argmax(mnt_w_out, axis=0)
        mnt_h_out = np.argmax(mnt_h_out, axis=0)
        mnt_o_out = np.argmax(mnt_o_out, axis=0)  # TODO: use ori_highest_peak(np version)
        # get final mnt
        mnt_final = np.zeros((len(mnt_list), 4))
        mnt_final[:, 0] = mnt_sparse.col * 8 + mnt_w_out[mnt_list[:, 0], mnt_list[:, 1]]
        mnt_final[:, 1] = mnt_sparse.row * 8 + mnt_h_out[mnt_list[:, 0], mnt_list[:, 1]]
        mnt_final[:, 2] = (mnt_o_out[mnt_list[:, 0], mnt_list[:, 1]] * 2 - 89.) / 180 * np.pi
        mnt_final[mnt_final[:, 2] < 0.0, 2] = mnt_final[mnt_final[:, 2] < 0.0, 2] + 2 * np.pi
        mnt_final[:, 3] = mnt_s_out[mnt_list[:, 0], mnt_list[:, 1]]
        return mnt_final

    def nms(self, mnt):
        if mnt.shape[0] == 0:
            return mnt
        # sort score
        mnt_sort = mnt.tolist()
        mnt_sort.sort(key=lambda x: x[3], reverse=True)
        mnt_sort = np.array(mnt_sort)
        # cal distance
        inrange = self.distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi / 6).astype(np.float32)
        keep_list = np.ones(mnt_sort.shape[0])
        for i in range(mnt_sort.shape[0]):
            if keep_list[i] == 0:
                continue
            keep_list[i + 1:] = keep_list[i + 1:] * (1 - inrange[i, i + 1:])
        return mnt_sort[keep_list.astype(np.bool), :]

    def angle_delta(self, A, B, max_D=np.pi * 2):
        delta = np.abs(A - B)
        delta = np.minimum(delta, max_D - delta)
        return delta

    def distance(self, y_true, y_pred, max_D=16, max_O=np.pi / 6):
        D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
        O = spatial.distance.cdist(np.reshape(y_true[:, 2], [-1, 1]), np.reshape(y_pred[:, 2], [-1, 1]), self.angle_delta)
        return (D <= max_D) * (O <= max_O)


def convert_to_gray(img):
    img_gray = img[:, 0, ...] * 0.299 + img[:, 1, ...] * 0.586 + img[:, 2, ...] * 0.114
    img_gray = img_gray[:, np.newaxis, ...]

    return img_gray


# image normalization
def img_normalization_torch(img_input, m0=0.0, var0=1.0):
    m = torch.mean(img_input)
    var = torch.var(img_input)
    after = ((var0 * (img_input - m) ** 2 / var) + 1e-07) ** 0.5
    image_n = torch.where(img_input > m, m0 + after, m0 - after)
    return image_n


# atan2 function
def atan2_torch(y_x):
    y, x = y_x[0], y_x[1] + 1e-07
    atan = torch.atan2(y,  x)
    return atan


def reduce_sum_torch(x):
    return torch.sum(x, 1, keepdim=True)


def merge_concat_torch(x):
    return torch.cat((x[0], x[1]), 1)


def select_max_torch(x):
    x_norm = x / (torch.max(x, 1, keepdim=True)[0] + 1e-07)
    x_norm = torch.where(x_norm > 0.999, x_norm, torch.zeros_like(x_norm))
    x_norm = x_norm / (torch.sum(x_norm, 1, keepdim=True) + 1e-07)  # prevent two or more ori is selected
    return x_norm


class FingerNet(nn.Module):
    def __init__(self, requires_grad=False):
        super(FingerNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 3, stride=1, dilation=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_1.training = False
        self.prelu1_1 = nn.PReLU(num_parameters=64, init=0.0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, dilation=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.bn1_2.training = False
        self.prelu1_2 = nn.PReLU(num_parameters=64, init=0.0)
        self.max_pooling2d_1 = nn.MaxPool2d((2, 2), stride=2)
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, dilation=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_1.training = False
        self.prelu2_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, dilation=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.bn2_2.training = False
        self.prelu2_2 = nn.PReLU(num_parameters=128, init=0.0)
        self.max_pooling2d_2 = nn.MaxPool2d((2, 2), stride=2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, dilation=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_1.training = False
        self.prelu3_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, dilation=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.bn3_2.training = False
        self.prelu3_2 = nn.PReLU(num_parameters=256, init=0.0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, dilation=1, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.bn3_3.training = False
        self.prelu3_3 = nn.PReLU(num_parameters=256, init=0.0)
        self.max_pooling2d_3 = nn.MaxPool2d((2, 2), stride=2)
        # scale 1
        self.conv4_1 = nn.Conv2d(256, 256, 3, stride=1, dilation=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.bn4_1.training = False
        self.prelu4_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.convori_1_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnori_1_1 = nn.BatchNorm2d(128)
        self.bnori_1_1.training = False
        self.preluori_1_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.ori_1_2 = nn.Conv2d(128, 90, 1, stride=1, dilation=1)
        self.convseg_1_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnseg_1_1 = nn.BatchNorm2d(128)
        self.bnseg_1_1.training = False
        self.preluseg_1_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.seg_1_2 = nn.Conv2d(128, 1, 1, stride=1, dilation=1)
        # scale 2
        self.atrousconv4_2 = nn.Conv2d(256, 256, 3, stride=1, dilation=4, padding=4)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.bn4_2.training = False
        self.prelu4_2 = nn.PReLU(num_parameters=256, init=0.0)
        self.convori_2_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnori_2_1 = nn.BatchNorm2d(128)
        self.bnori_2_1.training = False
        self.preluori_2_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.ori_2_2 = nn.Conv2d(128, 90, 1, stride=1, dilation=1)
        self.convseg_2_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnseg_2_1 = nn.BatchNorm2d(128)
        self.bnseg_2_1.training = False
        self.preluseg_2_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.seg_2_2 = nn.Conv2d(128, 1, 1, stride=1, dilation=1)
        # scale 3
        self.atrousconv4_3 = nn.Conv2d(256, 256, 3, stride=1, dilation=8, padding=8)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.bn4_3.training = False
        self.prelu4_3 = nn.PReLU(num_parameters=256, init=0.0)
        self.convori_3_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnori_3_1 = nn.BatchNorm2d(128)
        self.bnori_3_1.training = False
        self.preluori_3_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.ori_3_2 = nn.Conv2d(128, 90, 1, stride=1, dilation=1)
        self.convseg_3_1 = nn.Conv2d(256, 128, 1, stride=1, dilation=1)
        self.bnseg_3_1 = nn.BatchNorm2d(128)
        self.bnseg_3_1.training = False
        self.preluseg_3_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.seg_3_2 = nn.Conv2d(128, 1, 1, stride=1, dilation=1)
        # ----------------------------------------------------------------------------
        # enhance part
        self.enh_img_real_1 = nn.Conv2d(1, 90, 25, stride=1, dilation=1, padding=12)
        self.enh_img_imag_1 = nn.Conv2d(1, 90, 25, stride=1, dilation=1, padding=12)
        self.ori_peak = nn.Conv2d(90, 90, 1, stride=1, dilation=1, padding=0, bias=False)
        # ----------------------------------------------------------------------------
        # mnt part
        self.convmnt_1_1 = nn.Conv2d(2, 64, 9, stride=1, dilation=1, padding=4)
        self.bnmnt_1_1 = nn.BatchNorm2d(64)
        self.bnmnt_1_1.training = False
        self.prelumnt_1_1 = nn.PReLU(num_parameters=64, init=0.0)
        self.max_pooling2d_4 = nn.MaxPool2d((2, 2), stride=2)
        self.convmnt_2_1 = nn.Conv2d(64, 128, 5, stride=1, dilation=1, padding=2)
        self.bnmnt_2_1 = nn.BatchNorm2d(128)
        self.bnmnt_2_1.training = False
        self.prelumnt_2_1 = nn.PReLU(num_parameters=128, init=0.0)
        self.max_pooling2d_5 = nn.MaxPool2d((2, 2), stride=2)
        self.convmnt_3_1 = nn.Conv2d(128, 256, 3, stride=1, dilation=1, padding=1)
        self.bnmnt_3_1 = nn.BatchNorm2d(256)
        self.bnmnt_3_1.training = False
        self.prelumnt_3_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.max_pooling2d_6 = nn.MaxPool2d((2, 2), stride=2)

        self.convmnt_o_1_1 = nn.Conv2d(346, 256, 1, stride=1, dilation=1)
        self.bnmnt_o_1_1 = nn.BatchNorm2d(256)
        self.bnmnt_o_1_1.training = False
        self.prelumnt_o_1_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.mnt_o_1_2 = nn.Conv2d(256, 180, 1, stride=1, dilation=1)

        self.convmnt_w_1_1 = nn.Conv2d(256, 256, 1, stride=1, dilation=1)
        self.bnmnt_w_1_1 = nn.BatchNorm2d(256)
        self.bnmnt_w_1_1.training = False
        self.prelumnt_w_1_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.mnt_w_1_2 = nn.Conv2d(256, 8, 1, stride=1, dilation=1)

        self.convmnt_h_1_1 = nn.Conv2d(256, 256, 1, stride=1, dilation=1)
        self.bnmnt_h_1_1 = nn.BatchNorm2d(256)
        self.bnmnt_h_1_1.training = False
        self.prelumnt_h_1_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.mnt_h_1_2 = nn.Conv2d(256, 8, 1, stride=1, dilation=1)

        self.convmnt_s_1_1 = nn.Conv2d(256, 256, 1, stride=1, dilation=1)
        self.bnmnt_s_1_1 = nn.BatchNorm2d(256)
        self.bnmnt_s_1_1.training = False
        self.prelumnt_s_1_1 = nn.PReLU(num_parameters=256, init=0.0)
        self.mnt_s_1_2 = nn.Conv2d(256, 1, 1, stride=1, dilation=1)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input, label_nc=3):
        if label_nc == 3:
            input = convert_to_gray(input)

        bn_input = img_normalization_torch(input)
        out = self.conv1_1(bn_input)
        out = self.bn1_1(out)
        out = self.prelu1_1(out)
        out = self.conv1_2(out)
        out = self.bn1_2(out)
        out = self.prelu1_2(out)
        out = self.max_pooling2d_1(out)
        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.prelu2_1(out)
        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.prelu2_2(out)
        out = self.max_pooling2d_2(out)
        out = self.conv3_1(out)
        out = self.bn3_1(out)
        out = self.prelu3_1(out)
        out = self.conv3_2(out)
        out = self.bn3_2(out)
        out = self.prelu3_2(out)
        out = self.conv3_3(out)
        out = self.bn3_3(out)
        out = self.prelu3_3(out)
        out = self.max_pooling2d_3(out)
        # scale 1
        scale1 = self.conv4_1(out)
        scale1 = self.bn4_1(scale1)
        scale1 = self.prelu4_1(scale1)
        ori_1 = self.convori_1_1(scale1)
        ori_1 = self.bnori_1_1(ori_1)
        ori_1 = self.preluori_1_1(ori_1)
        ori_1 = self.ori_1_2(ori_1)
        seg_1 = self.convseg_1_1(scale1)
        seg_1 = self.bnseg_1_1(seg_1)
        seg_1 = self.preluseg_1_1(seg_1)
        seg_1 = self.seg_1_2(seg_1)
        # scale 2
        scale2 = self.atrousconv4_2(out)
        scale2 = self.bn4_2(scale2)
        scale2 = self.prelu4_2(scale2)
        ori_2 = self.convori_2_1(scale2)
        ori_2 = self.bnori_2_1(ori_2)
        ori_2 = self.preluori_2_1(ori_2)
        ori_2 = self.ori_2_2(ori_2)
        seg_2 = self.convseg_2_1(scale2)
        seg_2 = self.bnseg_2_1(seg_2)
        seg_2 = self.preluseg_2_1(seg_2)
        seg_2 = self.seg_2_2(seg_2)
        # scale 2
        scale3 = self.atrousconv4_3(out)
        scale3 = self.bn4_3(scale3)
        scale3 = self.prelu4_3(scale3)
        ori_3 = self.convori_3_1(scale3)
        ori_3 = self.bnori_3_1(ori_3)
        ori_3 = self.preluori_3_1(ori_3)
        ori_3 = self.ori_3_2(ori_3)
        seg_3 = self.convseg_3_1(scale3)
        seg_3 = self.bnseg_3_1(seg_3)
        seg_3 = self.preluseg_3_1(seg_3)
        seg_3 = self.seg_3_2(seg_3)

        ori_out = ori_1 + ori_2 + ori_3
        ori_out_1 = nn.Sigmoid()(ori_out)
        ori_out_2 = nn.Sigmoid()(ori_out)
        seg_out = seg_1 + seg_2 + seg_3
        seg_out = nn.Sigmoid()(seg_out)

        filter_img_real = self.enh_img_real_1(input)
        filter_img_imag = self.enh_img_imag_1(input)

        ori_peak = self.ori_peak(ori_out_1)
        ori_peak = select_max_torch(ori_peak)
        upsample_ori = nn.UpsamplingNearest2d(size=(ori_peak.shape[2] * 8, ori_peak.shape[3] * 8))(ori_peak)
        seg_round = nn.Softsign()(seg_out)
        upsample_seg = nn.UpsamplingNearest2d(size=(seg_round.shape[2] * 8, seg_round.shape[3] * 8))(seg_round)
        mul_mask_real = filter_img_real * upsample_ori
        enh_img_real = reduce_sum_torch(mul_mask_real)
        mul_mask_imag = filter_img_imag * upsample_ori
        enh_img_imag = reduce_sum_torch(mul_mask_imag)
        enh_img = atan2_torch([enh_img_imag, enh_img_real])
        enh_seg_img = merge_concat_torch([enh_img, upsample_seg])
        # ----------------------------------------------------------------------------
        # mnt part
        mnt_conv = self.convmnt_1_1(enh_seg_img)
        mnt_conv = self.bnmnt_1_1(mnt_conv)
        mnt_conv = self.prelumnt_1_1(mnt_conv)
        mnt_conv = self.max_pooling2d_4(mnt_conv)
        mnt_conv = self.convmnt_2_1(mnt_conv)
        mnt_conv = self.bnmnt_2_1(mnt_conv)
        mnt_conv = self.prelumnt_2_1(mnt_conv)
        mnt_conv = self.max_pooling2d_4(mnt_conv)
        mnt_conv = self.convmnt_3_1(mnt_conv)
        mnt_conv = self.bnmnt_3_1(mnt_conv)
        mnt_conv = self.prelumnt_3_1(mnt_conv)
        mnt_conv = self.max_pooling2d_4(mnt_conv)

        mnt_o_1 = merge_concat_torch([mnt_conv, ori_out_1])
        mnt_o_2 = self.convmnt_o_1_1(mnt_o_1)
        mnt_o_2 = self.bnmnt_o_1_1(mnt_o_2)
        mnt_o_2 = self.prelumnt_o_1_1(mnt_o_2)
        mnt_o_3 = self.mnt_o_1_2(mnt_o_2)
        mnt_o_out = nn.Sigmoid()(mnt_o_3)

        mnt_w_1 = self.convmnt_w_1_1(mnt_conv)
        mnt_w_1 = self.bnmnt_w_1_1(mnt_w_1)
        mnt_w_1 = self.prelumnt_w_1_1(mnt_w_1)
        mnt_w_2 = self.mnt_w_1_2(mnt_w_1)
        mnt_w_out = nn.Sigmoid()(mnt_w_2)

        mnt_h_1 = self.convmnt_h_1_1(mnt_conv)
        mnt_h_1 = self.bnmnt_h_1_1(mnt_h_1)
        mnt_h_1 = self.prelumnt_h_1_1(mnt_h_1)
        mnt_h_2 = self.mnt_h_1_2(mnt_h_1)
        mnt_h_out = nn.Sigmoid()(mnt_h_2)

        mnt_s_1 = self.convmnt_s_1_1(mnt_conv)
        mnt_s_1 = self.bnmnt_s_1_1(mnt_s_1)
        mnt_s_1 = self.prelumnt_s_1_1(mnt_s_1)
        mnt_s_2 = self.mnt_s_1_2(mnt_s_1)
        mnt_s_out = nn.Sigmoid()(mnt_s_2)

        return ori_out_2, upsample_ori, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out


def _test_fingernet_loss(image):
    # input
    img_size = image.shape
    img_size = np.array(img_size, dtype=np.int32) // 8 * 8
    image = image / 255.0
    image = image[:img_size[0], :img_size[1]]
    image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
    image = image.astype(np.float32)
    # run pytorch model
    pytorch_input = torch.from_numpy(image.transpose(0, 3, 1, 2))
    pytorch_input = pytorch_input.to('cuda')

    finger_net_loss = FingerNetLoss()
    finger_net_loss.forward(pytorch_input, pytorch_input)
