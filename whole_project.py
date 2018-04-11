#-*- coding:utf-8 _*-

'''
整个工程
'''
from classifier_TRAIN_API import mode_train
from classifier_CLASSIFIER_API import mode_classifier
from Faceplusplus_face_detect_API_s import faceplusplus_face_detect_api_s




data_dir = 'D:/Anaconda3/Lib/site-packages/facenet/data/lfw/lfw_align_mtcnnpy_160'

# 训练分类器
train_set_paths, train_set_true_labels, train_set_true_class_names = \
    mode_train(data_dir=data_dir, use_split_dataset=True, seed=666)

# 测试分类器
test_set_paths, test_set_true_labels, test_set_true_class_names = \
    mode_classifier(data_dir=data_dir, use_split_dataset=True, seed=666)

# 对测试集图像附加眼镜,返回镜框点集坐标，图片子路径，图片文件夹路径
store_point, face_glass_path, pre_dir = faceplusplus_face_detect_api_s(test_set_paths, test_set_true_class_names)

# 对戴黑色眼睛的图像测试精度
test_glass_face_paths, test_glass_face_labels, test_glass_face_class_names = \
    mode_classifier(data_dir=pre_dir, use_split_dataset=False, seed=666)
