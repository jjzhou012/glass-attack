'''
继续上一次迭代  接口
'''
#-*- coding:utf-8 _*-
from Faceplusplus_face_detect_API import faceplusplus_face_detect_api
#from Faceplusplus_face_detect_API_s import faceplusplus_face_detect_api_s
#from contributed.predict_API import predict_api
#from Dir_make import mk_dir
import numpy as np
#import random
#import time
from PIL import Image
#import cv2
#import os


def continue_last_iterator(filepath, particle_image, seed_useful, class_name):

    # filepath = ['D:/Anaconda3/Lib/site-packages/facenet/data/lfw/lfw_align_mtcnnpy_160\\Ariel_Sharon\\Ariel_Sharon_0041.png']
    # label = [0]
    # class_name = ['Ariel_Sharon']
    # 粒子数
    # particle_num = 3

    # 继续迭代的粒子数
    particle_num = len(particle_image)

    # 获取镜框坐标, 原始墨镜图像路径（str）
    store_point, face_glass_path = faceplusplus_face_detect_api(filepath[0], class_name[0])
    # 像素点数量
    pixel_num = len(store_point)

    #
    X = np.zeros((particle_num, pixel_num, 3))

    # 粒子图像反解析
    print('开始反解析...获取X...')
    for i in range(particle_num):
        print('第' + str(i+1) + '张...')
        # 读取 粒子图像
        print('读取第' + str(i+1) + '张粒子图像...')
        particle_glass_image = np.array(Image.open(particle_image[i]))
        for a in range(pixel_num):          # 第几个像素点
            X[i][a] = particle_glass_image[store_point[a][0]][store_point[a][1]]

    print('反解析结束!')

    #
    return particle_image, X, seed_useful, store_point



