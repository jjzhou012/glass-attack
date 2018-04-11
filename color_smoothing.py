#-*- coding:utf-8 _*-

from PIL import Image
import numpy as np
# import math
import time


def color_smoothing(particle_image_path, store_point):

    # st = time.time()

    face = np.array(Image.open(particle_image_path))
    VT = 0

    for i in range(len(store_point)):
        if [store_point[i][0], store_point[i][1] + 1] in store_point:
            if [store_point[i][0] + 1, store_point[i][1]] in store_point:
                #
                # print(face[store_point[i][0]][store_point[i][1]])
                # print(face[store_point[i][0] + 1][store_point[i][1]])
                # print(face[store_point[i][0]][store_point[i][1] + 1])
                r = sum(face[store_point[i][0]][store_point[i][1]]) / 3
                r_right = sum(face[store_point[i][0] + 1][store_point[i][1]]) / 3
                r_down = sum(face[store_point[i][0]][store_point[i][1] + 1]) / 3
                #
                delta_1 = np.square(r - r_right)
                delta_2 = np.square(r - r_down)
                result = np.sqrt(delta_1 + delta_2)

                VT += result

    VT = VT / len(store_point)
    # print(time.time() - st)

    return VT
