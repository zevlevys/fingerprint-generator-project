import scipy.misc
import imageio
import numpy as np
from PIL import Image
import os

def imread(path):
    return Image.open(path).convert('L')


def get_image(x, left, top, right, bottom):
    # print(image.size)
    # input_height, input_width = image.size
    resize_h = 512
    resize_w = 512
    x = x.crop((left, top, right, bottom))
    x = x.resize((resize_h, resize_w))
    return x


def parse_minute_file(mnt_file_path):
    try:
        mnt = np.loadtxt(mnt_file_path)[:, :4]
    except:
        return []
    mnt[:, -1] = (360 - mnt[:, -1]) * np.pi / 180
    return mnt


def main():
    base_path = '/home/zev/projects/OAI/older-fingerprint-repo/'
    for file in os.listdir(base_path + 'test/')[:1000]:
        # print(file)
        image = imread(base_path + 'test/' + file)
        w,h = image.size
        mnt = parse_minute_file(base_path + 'test_mintxt/' + file[:-4] + '.txt')
        if mnt == []:
            continue
        max_w = min(w, mnt[:, 1].max() + int(w/6))
        min_w = max(0, mnt[:, 1].min() - int(w/6))
        max_h = min(h, mnt[:, 2].max() + int(h/6))
        min_h = max(0, mnt[:, 2].min() - int(h/6))
        if max_w-min_w > max_h-min_h:
            diff = (max_w-min_w) - (max_h-min_h)
            max_h = min(h, max_h + int(diff / 2))
            min_h = max(0, min_h - int(diff / 2))
        elif max_w-min_w < max_h-min_h:
            diff = (max_h-min_h) - (max_w-min_w)
            max_w = min(w, max_w + int(diff / 2))
            min_w = max(0, min_w - int(diff / 2))

        left, top, right, bottom = (min_w, min_h, max_w, max_h)
        new_image = get_image(image, left, top, right, bottom)

        imageio.imsave(base_path + 'test_cropped/cropped-' + file[:-4] + '.png', new_image)


if __name__ == '__main__':
    main()