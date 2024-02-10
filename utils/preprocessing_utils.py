import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as sc
from skimage.draw import line_nd
from tqdm import tqdm


def list_files_in_folder(folder, with_path=True):
    # if folder contains folders of files
    folders = [os.path.join(folder, f) for f in os.listdir(folder)]
    if not os.path.isfile(folders[0]):
        files = []
        for folder in folders:
            if os.path.isdir(folder):
                if with_path:
                    files += [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
                else:
                    files += [f for f in os.listdir(folder) if f.endswith('.tif')]
        return files

    # else if folder contains files
    if with_path:
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]

    return [f for f in os.listdir(folder) if f.endswith('.tif')]


def parse_minute_file(mnt_file_path):
    mnt = np.loadtxt(mnt_file_path)[:, :4]
    mnt[:, -1] = (360 - mnt[:, -1]) * np.pi / 180
    return mnt


def parse_minute_file_old(mnt_file_path):
    with open(mnt_file_path, 'r') as mnt_file:
        mnt_list = [line.strip().split(' ') for line in mnt_file.readlines()]
        mnt_out = []
        for mnt in mnt_list:
            p1 = (int(mnt[1]), int(mnt[2]))
            p2 = (int(mnt[1]) + int(mnt[4]), int(mnt[2]) + int(mnt[4]))
            type = int(mnt[0])
            mnt_dict = {"type": type,
                        "p1": p1,
                        "p2": p2}
            mnt_out.append(mnt_dict)
        return mnt_out


def create_map_scipy(mnts, size=(768, 832), num_of_maps=3, ori_length=15, mnt_sigma=9, ori_sigma=3,
                     mnt_gain=60, ori_gain=3, include_singular=True):
    maps = []
    if include_singular:  # include core and delta points
        types = [[1], [2], [4, 5]]
    else:
        types = [[1], [2], [-1]]
    for idx in range(num_of_maps):
        _map = np.zeros(size)
        map_ori = np.zeros(size)
        for mnt_type in types[idx]:
            print(mnt_type)
            x = mnts[mnts[:, 0] == mnt_type, 1].astype(np.int32).tolist()
            print(x)
            y = mnts[mnts[:, 0] == mnt_type, 2].astype(np.int32).tolist()
            o = mnts[mnts[:, 0] == mnt_type, 3]
            # This line is a real fucker
            _map[(y, x)] = 0.5

            if mnt_type in [1, 2]:
                for x_l, y_l, o_l in zip(x, y, o):
                    print(x_l, y_l, o_l)
                    x1, x2, y1, y2 = (x_l, x_l + ori_length * np.cos(o_l), y_l, y_l + ori_length * np.sin(o_l))
                    line_idx = line_nd((y1, x1), (y2, x2), endpoint=True)
                    map_ori[line_idx] = 1

            if mnt_type in [4, 5]:
                mnt_sigma = 25
                mnt_gain = mnt_gain * 3
        map_blur = sc.gaussian_filter(_map, sigma=np.sqrt(mnt_sigma))[:, :, np.newaxis] * mnt_gain
        map_ori_blur = sc.gaussian_filter(map_ori, sigma=np.sqrt(ori_sigma))[:, :, np.newaxis] * ori_gain
        # maps.append(_map[:, :, np.newaxis] + map_ori[:, :, np.newaxis])
        maps.append(map_blur + map_ori_blur)
    output = np.concatenate(maps, axis=-1)
    output = output * 255
    output[output > 255] = 255
    output = output.astype('uint8')
    return output


def create_map_scipy_without_ori(mnt_dict_list, size=(768, 832), num_of_maps=3):
    maps = []
    types = [[1], [2], [4, 5]]
    for idx in range(num_of_maps):
        map = np.zeros(size)
        x = [mnt_dict['p1'][0] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        y = [mnt_dict['p1'][1] for mnt_dict in mnt_dict_list if mnt_dict['type'] in types[idx]]
        map[[y, x]] = 1
        maps.append(sc.gaussian_filter(map, sigma=np.sqrt(3))[:, :, np.newaxis] * 20)
    output = np.concatenate(maps, axis=-1)
    return output


def main():
    # input arguments
    include_singular = False
    imgs_path = '/home/zev/projects/OAI/fingerprint-generator/samples/tifs/'
    txts_path = '/home/zev/projects/OAI/fingerprint-generator/samples/txts/'
    output_path = '/home/zev/projects/OAI/fingerprint-generator/output/'

    # create output folder if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # iterate over minutiae text files and creates a corresponding maps
    imgs_name_list = list_files_in_folder(imgs_path, with_path=False)[:10]

    for idx, img_name in enumerate(tqdm(imgs_name_list)):
        try:
            mnt_file_path = os.path.join(txts_path, img_name.replace('tif', 'txt'))
            mnt = parse_minute_file(mnt_file_path)
            print(mnt)
            map = create_map_scipy(mnt, include_singular=include_singular, ori_gain=3, size=(300,300))
            plt.imsave(os.path.join(output_path, img_name[:-4] + '.png'), map)

        except IndexError:
            print("file currapted: {}".format(mnt_file_path))


if __name__ == "__main__":
    main()
