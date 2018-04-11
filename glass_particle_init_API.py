'''
单张图像 粒子图像生成    接口      随机像素眼镜+ 平滑眼镜（50%）                !!!服务器： 修改两处路径
'''
#-*- coding:utf-8 _*-
from Faceplusplus_face_detect_API import faceplusplus_face_detect_api
# from Faceplusplus_face_detect_API_s import faceplusplus_face_detect_api_s
# from predict_API import predict_api
from Dir_make import mk_dir
import numpy as np
import time
from PIL import Image
import os


def glass_particle_init_api(filepath, label, class_name, particle_num):  # 输入参数 list

    # filepath = ['D:/Anaconda3/Lib/site-packages/facenet/data/lfw/lfw_align_mtcnnpy_160\\Ariel_Sharon\\Ariel_Sharon_0041.png']
    # label = [0]
    # class_name = ['Ariel_Sharon']
    # 粒子数
    # particle_num = 3

    # 获取镜框坐标, 原始墨镜图像路径（str）
    store_point, face_glass_path = faceplusplus_face_detect_api(filepath[0], class_name[0])
    # 像素点数量
    pixel_num = len(store_point)
    # 随机像素堆  存放列表初始化
    glass_particle_pixel_list = []
    # 像素眼镜（粒子）图像 存放列表初始化
    glass_particle_path_list = []
    # 随机种子存放列表
    seed_list = []

    print('开始生成像素眼镜（particle）...')

    # 生成随机像素眼镜
    for i in range(particle_num):
        # 获取seed
        Time = time.time()
        time.sleep(3)
        # print(str(Time))
        # 生成随机数
        seed = int(Time)
        np.random.seed(seed)

        if i < particle_num * 0.5:                  # 一半眼镜随机噪声眼镜
            glass_particle = np.random.randint(256, size=(pixel_num, 3))
        else:                                       # 一般为纯色眼镜
            color = np.random.randint(256, size=(1, 3))
            glass_particle = np.zeros((pixel_num, 3))
            for k in range(pixel_num):
                glass_particle[k] = color

        # print(glass_particle)
        # 存放像素眼镜,seed
        glass_particle_pixel_list.append(glass_particle)        # 50 * (RGB)
        seed_list.append(seed)

    # print(glass_particle_pixel_list)
    # print(glass_particle_pixel_list[0])
    print('------example:  pixel_stack--------')
    print(glass_particle_pixel_list[0])
    print('-----------------------------------')
    # print(seed_list)

    # 读取 原始黑框眼镜图像
    print('读取原始黑框眼镜图像...')
    black_glass_face = np.array(Image.open(face_glass_path))

    # 眼睛渲染
    print('开始眼镜渲染...')
    for i in range(particle_num):
        print('第' + str(i+1) + '张...')
        for a in range(pixel_num):          # 第几个像素点
            black_glass_face[store_point[a][0]][store_point[a][1]] = glass_particle_pixel_list[i][a]

        glass_particle = Image.fromarray(black_glass_face)

        # 保存
        print('---save...')
        # 目录生成
        mk_dir('/root/facenet/data/glass_particle_image/' + class_name[0])      # fix........................
        start_index = filepath[0].find(class_name[0] + '_')
        # 路径生成
        glass_particle_path = '/root/facenet/data/glass_particle_image/' + class_name[0] + '/' + \
                          filepath[0][start_index: -4] + '_glass_particle_' + str(i) + '.png'
        glass_particle.save(glass_particle_path, 'png')
        # 路径存入列表
        glass_particle_path_list.append(glass_particle_path)

    print('渲染结束!')

    # 返回粒子图像路径列表， 眼镜像素列表， 随机种子列表
    return glass_particle_path_list, glass_particle_pixel_list, seed_list, store_point


def X_to_particle_image_path(filepath, store_point, X, particle_image_path):

    face = np.array(Image.open(filepath))   # filepath(str)
    pixel_num = len(store_point)

    for a in range(pixel_num):  # 第几个像素点
        face[store_point[a][0]][store_point[a][1]] = X[a]

    glass_particle = Image.fromarray(face)

    # 保存
    os.remove(particle_image_path[0])
    glass_particle.save(particle_image_path[0], 'png')

    return particle_image_path[0]
