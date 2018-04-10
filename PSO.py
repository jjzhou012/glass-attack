#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
 粒子群优化算法
 particle： 25个随机生成的像素图像，     未实现颜色平滑                     ！！！ 服务器： 修改三处路径

"""
from glass_particle_init_API import glass_particle_init_api
from glass_particle_init_API import X_to_particle_image_path
from predict_API import predict_api      # fix..............................
from continue_last_inerator import continue_last_iterator
from color_smoothing import color_smoothing
from Dir_make import mk_dir
import numpy as np
import matplotlib.pyplot as plt
import shutil, os

# ----------------------PSO参数设置---------------------------------
class PSO():
    print('PSO初始化...')
    def __init__(self, filepath, label, class_name, target, pN, max_iter, min_bl=True, continue_last = False, continue_file = None, continue_seed=None):
        self.w = 0.9
        # 权值线性递减
        #self.w = 0.9
        self.w_end = 0.4

        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN                                        # 粒子数量
        self.orig_filepath = filepath
        self.continue_last = continue_last                  # 是否继续上一次迭代
        self.continue_file = continue_file                  # 继续文件

        if self.continue_last == False:
            # 粒子初始化接口
            self.particle_image, self.glass_particle_pixel_list, self.seed_list, self.store_point = \
                glass_particle_init_api(filepath=filepath, label=label, class_name=class_name, particle_num=pN)

        else:
            self.particle_image, self.continue_X, self.seed_list, self.store_point = continue_last_iterator(filepath=filepath,
                                        particle_image=continue_file, seed_useful=continue_seed, class_name=class_name)
        # 目标攻击的标签(str)
        self.target = target
        print('冒充目标：' + target)
        # 原始图像 top-3
        self.original_top_three, self.orig_top_three_class_name = predict_api(filepath)
        print(self.original_top_three)
        print(self.orig_top_three_class_name)
        # 判断target是否在原始top-3中, 若不在，则以第二target作为当前目标
        if self.target not in self.orig_top_three_class_name:
            self.target_current = self.orig_top_three_class_name[1]
        else:
            self.target_current = target
        print('当前目标：' + self.target_current)

        self.seed_useful = []                               # 保存较好的seed
        self.seed_mid = []                                  # 过渡seed
        self.particle_useful = []                           # 保存较好的粒子
        self.particle_mid = []                              # 保存过渡粒子

        self.k_1 = 0.15                                     # 平滑系数

        self.dim = len(self.store_point)                    # 搜索维度
        self.max_iter = max_iter                            # 迭代次数
        self.min_bl = min                                   # 是否为最小化适应度函数
        self.X = np.zeros((self.pN, self.dim, 3))           # 所有粒子的位置和速度
        print(self.X)
        self.V = np.zeros((self.pN, self.dim, 3))
        self.pbest = np.zeros((self.pN, self.dim, 3))       # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim, 3))
        self.p_fit = np.zeros(self.pN)                      # 每个个体的历史最佳适应值
        self.fit = 50                                       # 全局最佳适应值
        self.max_object = 40                                # 劣值

        self.init_Population()                              # 初始化种群


# ---------------------初始化种群----------------------------------
    def init_Population(self):
        print('粒子群初始化...')
        if self.continue_last == False:
            print('新的PSO迭代...初始化...')
            for i in range(self.pN):
                print('初始化粒子' + str(i+1))
                for j in range(self.dim):
                    # 完全初始化
                    self.X[i][j] = self.glass_particle_pixel_list[i][j]
                    self.V[i][j] = np.random.uniform(0, 20, size=(1, 3))
                self.pbest[i] = self.X[i]
                tmp = self.function([self.particle_image[i]], self.seed_list[i])
                self.p_fit[i] = tmp
                if(tmp < self.fit):
                    self.fit = tmp
                    self.gbest = self.X[i]
            print('初始化完毕' + '\n')
        else:
            print('继续上一次的迭代...初始化...')
            for i in range(self.pN):
                print('初始化粒子' + str(i + 1))
                for j in range(self.dim):
                    # 初始化
                    self.X[i][j] = self.continue_X[i][j]
                    self.V[i][j] = np.random.uniform(0, 10, size=(1, 3))
                self.pbest[i] = self.X[i]
                tmp = self.function([self.particle_image[i]], self.seed_list[i])
                self.p_fit[i] = tmp
                if(tmp < self.fit):
                    self.fit = tmp
                    self.gbest = self.X[i]
            print('初始化完毕' + '\n')

# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        print('\n' + '开始粒子更新...')
        fitness = []
        for t in range(self.max_iter):
            print('%s %s %s' % ('第', str(t+1), '次迭代...'))
            report = []
            # 权值更新
            self.w = (self.w - self.w_end) * (self.max_iter - t - 1) / self.max_iter + self.w_end

            for i in range(self.pN):         # 更新gbest\pbest
               print('\n' + '迭代序列：' + str(t+1) + '--更新粒子' + str(i+1))
               temp = self.function([self.particle_image[i]], self.seed_list[i])
               if(temp < self.p_fit[i]):      # 更新个体最优
                   self.p_fit[i] = temp
                   self.pbest[i] = self.X[i]
                   if(self.p_fit[i] < self.fit):  # 更新全局最优
                       self.gbest = self.X[i]
                       self.fit = self.p_fit[i]
               # 一次迭代的所有粒子的fit
               report.append(temp)
               
            index = np.argsort(report)   # 从小到大排序
            # save best image in per iterator
            mk_dir('D:/Anaconda3/Lib/site-packages/facenet/data/PSO_iterator/' + class_name[0])       # fix..............................
            shutil.copy(self.particle_image[index[0]], 'D:/Anaconda3/Lib/site-packages/facenet/data/PSO_iterator/'
                          + class_name[0] + '/' + 'iterator_' + str(t+1) + '_particle_' + str(index[0] + 1) + '.png')

            for i in range(self.pN):
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i])\
                       + self.c2*self.r2*(self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                # 位置映射到原图，返回更新粒子的路径
                self.particle_image[i] = X_to_particle_image_path(filepath=self.orig_filepath[0],
                    store_point=self.store_point, X=self.X[i], particle_image_path=[self.particle_image[i]])

            fitness.append([report - 1, self.fit])
            print("%s  %.4f" % ('当前第' + str(index[0] + 1) + '粒子达到全局最优：', self.fit))                # 输出最优值
            print('\n')

        return fitness, self.target_current, self.seed_useful, self.particle_useful

# ---------------------目标函数     自定义函数-----------------------------
    def function(self, particle_image_path, seed):
        #
        print('当前目标：' + self.target_current)

        # 传入参数为 粒子图
        top_three, top_three_class_name = predict_api(particle_image_path)  # 返回前三（类：置信度）对， 字典形式[{'class_name': , 'score': }]
        # 如果target出现在top-3，保存seed,
        if self.target in top_three_class_name:
            print('case1 : target in top_three')

            func = (top_three_class_name.index(self.target) + 1) * top_three[0]['score'] / top_three[top_three_class_name.index(self.target)]['score']
            # 平滑性
            VT = color_smoothing(particle_image_path=particle_image_path[0], store_point=self.store_point)
            print('VT:   ' + str(VT))
            # fit
            fit = func + self.k_1 * VT

            # 保存较好的seed,粒子图
            if seed not in self.seed_useful:
                self.seed_useful.append(seed)
                self.particle_useful.append(particle_image_path[0])
            # 全局更新target
            self.target_current = self.target
            print('fit:   ' + str(fit))

            return fit
        # 如果未出现target, 保存第二高置信度的target的seed，
        # 若之后出target出现在top-3，则清空该中间寄存器
        elif self.target_current in top_three_class_name:
            print('case2 ')
            # 判断是否出现新的候选目标, 若出现，取代成为新的target_current
            count = 0
            for i in range(3):
                for j in range(3):
                    if top_three_class_name[i] != self.orig_top_three_class_name[j]:
                        count += 1
                if count == 3:
                    # 取代
                    self.target_current = top_three_class_name[i]
                    print('case2:  出现新的候选，更新中间目标')
                    # 清空
                    self.seed_mid.clear()
                    break
                else:
                    count = 0

            # 计算fit
            func = (top_three_class_name.index(self.target_current) + 1) * top_three[0]['score'] / top_three[top_three_class_name.index(self.target_current)]['score']
            # 平滑性
            VT = color_smoothing(particle_image_path=particle_image_path[0], store_point=self.store_point)
            print('VT:   ' + str(VT))
            fit = func + self.k_1 * VT

            # 保存中间seed
            if seed not in self.seed_mid:
                self.seed_mid.append(seed)
                self.particle_mid.append(particle_image_path[0])

            print('fit:   ' + str(fit))
            return fit

        else:
            print('case3:  第二置信的类作为target_current')
            self.target_current = self.orig_top_three_class_name[1]

            VT = color_smoothing(particle_image_path=particle_image_path[0], store_point=self.store_point)
            print('VT:   ' + str(VT))
            fit = self.max_object + self.k_1 * VT
            print('fit:  ' + str(fit))
            return fit


# ----------------------绘图----------------------------------
    def ShowData(self,pltarg,line=["b",3]):
        plt.figure(1)
        plt.title(pltarg[0])
        plt.xlabel("iterators", size=pltarg[1][0])
        plt.ylabel("fitness", size=pltarg[1][1])
        t = np.array([t for t in range(pltarg[2][0],pltarg[2][1])])
        fitness = np.array(self.iterator())
        plt.plot(t, self.iterator(), color=line[0], linewidth=line[1])
        plt.show()

        return self.fit

# ----------------------获取数据----------------------------------
    def GetResult(self, pltarg =["Figure1", [14, 14], [0, 100]] ,line = ["b",3],show_bl = True,):
        if show_bl == True:
            return self.ShowData(pltarg, line=["b", 3])
        else:
            return self.iterator()

# ----------------------定义参数-----------------------

pltarg = ["Figure1",[14,14],[0,100]]
line = ["b", 3]

filepath = ['D:/Anaconda3/Lib/site-packages/facenet/data/lfw/lfw_align_mtcnnpy_160\\Ariel_Sharon\\Ariel_Sharon_0041.png']   # fix..............................
label = [0]
class_name = ['Ariel_Sharon']   #  有下划线，注意!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 粒子数
particle_num = 50
max_iter = 100


# ----------------------程序执行---输出结果----------------------------------
#
'''
test = PSO_iterator(filepath=filepath, label=label, class_name=class_name, target='Colin Powell',
           pN=particle_num, max_iter=max_iter, continue_last=False, continue_file=None, continue_seed=None)

# test.init_Population()

fitness, target_current, seed_useful, particle_useful = test.iterator()
print(fitness)
print(particle_useful)
print(seed_useful)

# 继续上一次
continue_particle_num = len(particle_useful)

test_continue = PSO_iterator(filepath=filepath, label=label, class_name=class_name, target='Colin Powell',
           pN=continue_particle_num, max_iter=max_iter, continue_last=True, continue_file=particle_useful, continue_seed=seed_useful)


fitness_2, target_current_2, seed_useful_2, particle_useful_2 = test_continue.iterator()
print(fitness_2)
print(particle_useful_2)
print(seed_useful_2)
'''

#

test = PSO(filepath=filepath, label=label, class_name=class_name, target='Colin Powell',
           pN=particle_num, max_iter=max_iter, continue_last=False, continue_file=None, continue_seed=None)

fitness, target_current, seed_useful, particle_useful = test.iterator()
print(fitness)
print(particle_useful)
print(seed_useful)


while fitness[-1][1] != 5:
    continue_particle_num = len(particle_useful)

    test_continue = PSO(filepath=filepath, label=label, class_name=class_name, target='Colin Powell',
                        pN=continue_particle_num, max_iter=max_iter, continue_last=True, continue_file=particle_useful,
                        continue_seed=seed_useful)
    fitness, target_current, seed_useful, particle_useful = test_continue.iterator()
    print(fitness)
    print(particle_useful)
    print(seed_useful)
