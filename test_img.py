# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
# np.set_printoptions(threshold='nan')

'''
filepath = "D:/Anaconda3/Lib/site-packages/facenet/data/test_image/1.png"

img = cv2.imread(filepath)


cv2.imshow('clean_image', img)
cv2.waitKey(1)
cv2.imwrite('1.jpg', img)

point_img = cv2.circle(img, (55, 68), 2, (0, 0, 255), -1)    # circle(图像，圆心，半径，颜色，填充)


cv2.imshow('adv_image', point_img)
cv2.waitKey(0)
'''
img = np.array(Image.open('glass.png'))  #打开图像并转化为数字矩阵

print(Image.open('glass.png'))


print(img[99][299] == [255, 255, 255])
print(np.size(img))
print(img.shape)
print(img.shape[1])

h, w, dim = img.shape

print(h)
print(w)

plt.figure("glass")
plt.imshow(img)
plt.axis('off')
plt.show()

