from itertools import product
from math import sqrt

import tensorflow as tf


def make_priors(img_size_h, img_size_w, feature_map_size, aspect_ratio, scale):
    """
    Create anchor boxes for each feature maps in [x, y, w, h], (x, y) is the center of anchor
    :param feature_map_size:
    :param img_size:
    :param aspect_ratio:
    :param scale:
    :return:
    """
    prior_boxes = []
    # num_anchors = []
    # for idx, f_size in enumerate(feature_map_size):
    #     # print("Create priors for f_size:%s", f_size)
    #     count_anchor = 0
    #     for j, i in product(range(int(f_size[0])), range(int(f_size[1]))):
    #         f_k_h = img_size_h / (f_size[0] + 1)
    #         f_k_w = img_size_w / (f_size[1] + 1)
    #         x = f_k_w * (i + 1)
    #         y = f_k_h * (j + 1)
    #         for ars in aspect_ratio:
    #             a = sqrt(ars)
    #             w = scale[idx] * a
    #             h = scale[idx] / a
    #             prior_boxes += [x - (w / 2), y - (h / 2), x + (w / 2), y + (h / 2)]
    #             count_anchor += 1
    #     num_anchors.append(count_anchor)
    #     # print(f_size, count_anchor)
    num_anchors = 0
    for idx, f_size in enumerate(feature_map_size):
        # print("Create priors for f_size:%s", f_size)
        count_anchor = 0
        for j, i in product(range(int(f_size[0])), range(int(f_size[1]))):
            # i,j are pixels values in feature map
            # + 0.5 because priors are in center
            x = (i + 0.5) / f_size[1] # normalize the pixel values
            y = (j + 0.5) / f_size[0]
            for ars in aspect_ratio:
                a = sqrt(ars)
                w = scale[idx] * a / img_size_w # normalize the width value
                h = scale[idx] / a / img_size_h
                # the author in original paper accidetly use square anchor for all the time
                # h = w
                # directly use point form here => [ymin, xmin, ymax, xmax]
                ymin = y - (h / 2)
                xmin = x - (w / 2)
                ymax = y + (h / 2)
                xmax = x + (w / 2)
                prior_boxes += [ymin * img_size_h, xmin * img_size_w, ymax * img_size_h, xmax * img_size_w]
            count_anchor += 1
        num_anchors += count_anchor
    output = tf.reshape(tf.convert_to_tensor(prior_boxes), [-1, 4])
    return num_anchors, output
