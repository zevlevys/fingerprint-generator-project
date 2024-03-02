import scipy.misc
import imageio
import numpy as np
from PIL import Image, ImageFile
import os
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True

def imread(path):
    return Image.open(path).convert('L')


def get_image(x, left, top, right, bottom):
    # print(image.size)
    # input_height, input_width = image.size
    resize_h = 256
    resize_w = 256
    x = x.crop((left, top, right, bottom))
    x = x.resize((resize_h, resize_w))
    return x


def parse_minute_file(mnt_file_path):
    try:
        mnt = np.loadtxt(mnt_file_path)[:, :4]
    except:
        return np.array([])
    mnt[:, -1] = (360 - mnt[:, -1]) * np.pi / 180
    return mnt


def main(dataset="train", base_path):
    # base_path = '/home/zev/projects/OAI/fingerprint-generator-project/fingerprint_dataset'
    print(f'{dataset}=, {base_path}=')
    
    i = 0
    corrupted = 0
    for file in os.listdir(base_path + f'{dataset}/'):
        # print(file)
        image = imread(base_path + f'{dataset}/' + file)
        w,h = image.size
        mnt = parse_minute_file(base_path + f'{dataset}_mintxt/' + file[:-4] + '.txt')
        if mnt.size == 0:
            corrupted += 1
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

        imageio.imsave(base_path + f'{dataset}_cropped/cropped-' + file[:-4] + '.png', new_image)
        i += 1
        if (i % 1000 == 0):
            print('Hit', i, 'images with', corrupted, 'corrupted images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test')
    parser.add_argument('-d', '--dataset') 
    parser.add_argument('-p', '--base_path')
    args = parser.parse_args()
    main(args.dataset, args.base_path)
