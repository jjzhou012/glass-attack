'''
face++ 眼镜定位，  镜框贴图   , 单张图片
'''
from __future__ import division

import requests
from json import JSONDecoder
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import misc
import os
from Dir_make import mk_dir
import time
import math

# 初始化
# 原始镜框信息
glass_filepath = 'D:/Anaconda3/Lib/site-packages/facenet/glass.png'
# 原始镜框定位点
# 定位点
glass_left = {'x': 80, 'y': 43}
glass_right = {'x': 220, 'y': 43}
# 眼镜对应间距
glass_point_distance = glass_right['x'] - glass_left['x']


def faceplusplus_face_detect_api(filepath, class_name):

    # 图片路径
    # filepath = "D:/Anaconda3/Lib/site-packages/facenet/data/test_image/1.png"

    # 原图显示
    clean_img = cv2.imread(filepath)
    cv2.imshow('clean_image', clean_img)
    cv2.waitKey(1)


    # 人眼检测
    left_eye_center, right_eye_center, angle = eye_detect(filepath)
    # print(str(left_eye_center))
    # print(str(right_eye_center))

    # 标注瞳孔
    point_img_left_eye = cv2.circle(clean_img, (left_eye_center['x'], left_eye_center['y']), 2, (0, 0, 255), -1)  # circle(图像，圆心，半径，颜色，填充)
    point_img = cv2.circle(point_img_left_eye, (right_eye_center['x'], right_eye_center['y']), 2, (0, 0, 255), -1)

    cv2.imshow('point_image', point_img)
    cv2.imwrite('point_image.png', point_img)
    cv2.waitKey(1)

    # 缩放比例计算，新镜框定位点
    new_glass_left, new_glass_right = scale(left_eye_center, right_eye_center)

    # 贴图, 并获取镜框坐标
    store_point, face = wear_glass_1(filepath, new_glass_left, left_eye_center, angle)

    # 数组转化为Image类
    face_glass = Image.fromarray(face)

    # 保存路径生成

    mk_dir('D:/Anaconda3/Lib/site-packages/facenet/data/face_glass_image/' + class_name)

    start_index = filepath.find(class_name + '_')

    face_glass_path = 'D:/Anaconda3/Lib/site-packages/facenet/data/face_glass_image/' + class_name + '/' + \
                      class_name + filepath[start_index: -4] + '_glass.png'
    face_glass.save(face_glass_path, 'png')

    return store_point, face_glass_path


def eye_detect(filepath):
    '''
    调用Face++ 人眼定位
    '''

    http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    key = "dHTpFppve9bV3ZqkvJadC74fQql3paRp"
    secret = "19R5whMXkjXyFCPEivqHPCrelkxfxByi"

    data = {"api_key": key, "api_secret": secret, "return_landmark": "1"}
    files = {"image_file": open(filepath, "rb")}
    # print(files)
    req_dict = {}

    #print(req_dict)
    # print(req_dict['faces'])
    #print(req_dict['faces'][0])
    #print(req_dict['faces'][0]['landmark'])
    # 远程调用防中断
    while 'faces' not in req_dict:
        response = requests.post(http_url, data=data, files=files)

        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)

    landmark_inf = req_dict['faces'][0]['landmark']
    left_eye_center = landmark_inf['left_eye_center']
    right_eye_center = landmark_inf['right_eye_center']

    #print('%s: %s' % ('left_eye_center', str(left_eye_center)))
    #print('%s: %s' % ('right_eye_center', str(right_eye_center)))

    # 求倾斜角度
    delta_x = left_eye_center['x'] - right_eye_center['x']
    delta_y = left_eye_center['y'] - right_eye_center['y']
    tan_ = delta_y / delta_x
    #print(tan_)
    angle = math.atan(tan_)

    return left_eye_center, right_eye_center, angle


def scale(left_eye_center, right_eye_center):
    '''
    获取缩放后的镜框定位点

    :param left_eye_center:
    :param right_eye_center:
    :param glass_point_distance:
    :return:
    '''
    # 瞳孔间距
    real_eye_distance = right_eye_center['x'] - left_eye_center['x']
    # 放缩比例
    k = real_eye_distance / glass_point_distance
    #print(k)

    # 镜框缩放
    glass = Image.open(glass_filepath)
    w, h = glass.size
    # glass_new = misc.imresize(glass, k)
    glass.thumbnail((int(w * h), int(h * k)))
    glass.save('glass_new.png', 'png')

    # 新镜框定位点
    new_glass_left = {}
    new_glass_right = {}

    # 定位点缩放
    new_glass_left['x'] = int(glass_left['x'] * k)
    new_glass_left['y'] = int(glass_left['y'] * k)
    new_glass_right['x'] = int(glass_right['x'] * k)
    new_glass_right['y'] = int(glass_right['y'] * k)

    return new_glass_left, new_glass_right


def wear_glass_1(filepath, new_glass_left, left_eye_center, angle):
    '''
    眼镜贴图方法一：
    根据左眼和镜框左定位点重合，进行像素替换

    :param new_glass_left:
    :return:
    '''
    # 存放镜框位置坐标
    store_point = []   # [i, j]
    pkg_point = []
    '''
    begin_point = {}
    begin_point['x'] = left_eye_center['x'] - new_glass_left['x']
    begin_point['y'] = left_eye_center['y'] - new_glass_left['y']
    '''
    face = np.array(Image.open(filepath))

    new_glass = np.array(Image.open('glass_new.png'))
    h, w, dim = new_glass.shape
    H, W, Dim = face.shape

    cos_ = math.cos(angle)
    sin_ = math.sin(angle)

    start_time = time.time()

    for i in range(w):
        for j in range(h):
            if (sum(new_glass[j][i]) <= 510):  # 若为镜框点
                pkg_point.append([j, i])
    for i in range(W):
        for j in range(H):
            vec_x_0 = i - left_eye_center['x']
            vec_y_0 = j - left_eye_center['y']
            vec_x = int(vec_x_0 * cos_ + vec_y_0 * sin_)
            vec_y = int(vec_x_0 * (-sin_) + vec_y_0 * cos_)
            if [(new_glass_left['y'] + vec_y), (new_glass_left['x'] + vec_x)] in pkg_point:
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        if 0<=j+b<=159 and 0<=i+a<=159 and (a*b == 0):
                            face[j+b][i+a] = [0, 0, 0]
                            if [j+b, i+a] not in store_point:
                                store_point.append([j+b, i+a])
    end_time = time.time()
    #print(end_time - start_time)
    print('%s: %f %s' % ('about', len(store_point) / (16 * 16), '%'))
    return store_point, face

'''
if __name__ == '__faceplusplus_face_detect_api__':
    faceplusplus_face_detect_api()
'''