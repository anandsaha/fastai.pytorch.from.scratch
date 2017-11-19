from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_image_size(path: object):
    img = Image.open(path)
    width, height = img.size
    return width, height


def print_stats(widths, heights, smallest_file, largest_file):
    print('Count: ', len(widths))

    print('Mean Width:', np.mean(widths))
    print('Min Width:', np.min(widths))
    print('Max Width:', np.max(widths))

    print('Mean Height:', np.mean(heights))
    print('Min Height:', np.min(heights))
    print('Max Height:', np.max(heights))

    print('Smallest file:', smallest_file)
    print('Largest File:', largest_file)

    plt.hist(widths, color='red', label='widths')
    plt.hist(heights, color='blue', label='heights')
    plt.legend()
    plt.show()

    plt.imshow(plt.imread(smallest_file))
    plt.show()

    plt.imshow(plt.imread(largest_file))
    plt.show()


def get_stats_from_path(path, ext='jpg'):
    widths = list()
    heights = list()

    smallest_file = ''
    largest_file = ''
    min_size = 9999999
    max_size = 0

    for d in os.listdir(path):
        d = os.path.join(path, d)
        print(d)
        if os.path.isdir(d):
            for f in os.listdir(d):
                f = os.path.join(d, f)
                if f.endswith(ext):
                    width, height = get_image_size(f)
                    widths.append(width)
                    heights.append(height)

                    if min_size > (height + width):
                        min_size = height + width
                        smallest_file = f
                    if max_size < (height + width):
                        max_size = height + width
                        largest_file = f

    print_stats(widths, heights, smallest_file, largest_file)


def get_stats_from_csv(csv_file, data_path, ext='jpg'):
    labels = pd.read_csv(csv_file).values
    widths = list()
    heights = list()

    smallest_file = ''
    largest_file = ''
    min_size = 9999999
    max_size = 0

    for l in labels:
        l[0] += '.' + ext
        path = os.path.join(data_path, l[0])
        width, height = get_image_size(path)
        widths.append(width)
        heights.append(height)

        if min_size > (height+width):
            min_size = height + width
            smallest_file = path
        if max_size < (height+width):
            max_size = height + width
            largest_file = path


    print_stats(widths, heights, smallest_file, largest_file)