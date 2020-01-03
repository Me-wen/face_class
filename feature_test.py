#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py


# In[2]:


test_path = "test"


# bins for histogram
bins = 30

# 特徵描述符-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# 特徵描述符-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# 特徵描述符-3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

glob_features = []



list_of_files = os.listdir(test_path)

    
for imag_file in list_of_files:
    imag_file_path = os.path.join(test_path,imag_file)

    # 讀取影像
    image_ = cv2.imread(imag_file_path)
    if image_ is None:
        raise RuntimeError("No image")
     
    # 特徵提取
    
    fv_hu_moments = fd_hu_moments(image_)
    fv_haralick   = fd_haralick(image_)
    fv_histogram  = fd_histogram(image_)

    
    # 串接功能+更新標籤和特徵向量列表
    
    glob_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    glob_features.append(glob_feature)


print(".... completed Feature Extraction of test data...")

# 獲得整體特徵向量的大小
print(".... feature vector size {}".format(np.array(glob_features).shape))

    
# 規範範圍（0-1）中的特徵向量
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features_ = scaler.fit_transform(glob_features)
print(".... feature vector normalized...")

# 使用HDF5保存特徵向量
h5f_test_data = h5py.File('Output/test_data.h5', 'w')
h5f_test_data.create_dataset('dataset_1', data=np.array(rescaled_features_))

h5f_test_data.close()

print(".... end of vectorisation of test data..")

