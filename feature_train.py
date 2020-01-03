#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py


# In[2]:


# path to training data
train_path="train"

#bns for histogram
bins=30

#特徵描述符1-Hu Moments
def fd_hu_moments(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    feature=cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#特徵描述符2-Haralick Texture
def fd_haralick(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    haralick=mahotas.features.haralick(gray).mean(axis=0)
    return haralick

#特徵描述符3-Color Histogram
def fd_histogram(image,mask=None):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([image],[0,1,2],None,[bins,bins,bins],[0,256,0,256,0,256])
    cv2.normalize(hist,hist)
    return hist.flatten()

#獲得Train標籤
train_labels=os.listdir(train_path)

#對Train Label進行排序
train_labels.sort()
print(train_labels)

#初始化列表以保存特徵向量和標籤
global_features=[]
labels=[]

#每個類別的圖片
images_per_class= 30

#loop over the training data sub-folders
for training_name in train_labels:
    #加入Train數據路徑和每個物種訓練文件夾
    dir_=os.path.join(train_path,training_name)
    
    #獲取當前的訓練標籤
    current_label=training_name
    
    list_of_files=os.listdir(dir_)
    # loop over the images in each sub-folder
    for img_file in list_of_files:
        img_file_path=os.path.join(train_path,current_label,img_file)
        
        #讀取影像
        image=cv2.imread(img_file_path)
        if image is None:
            raise RuntimeError("No image Found")
            
        #特徵提取
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        
        #串接功能
        global_feature=np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
        
        #更新標籤和特徵向量列表
        labels.append(current_label)
        global_features.append(global_feature)
        
        #處理文件夾
        print("... Processed Folder: {}".format(current_label))
        
#完成特徵提取
print("...Completed Feature Extraction of Training Data...") 

#獲得整體特徵向量的大小
print("...Feature vector size {}".format(np.array(global_features).shape))

#獲取整體訓練標籤的大小
print("...Training Labels {}".format(np.array(labels).shape))

#編碼目標標籤
targetNames=np.unique(labels)
le=LabelEncoder()
target=le.fit_transform(labels)
print(".....Training labels encoded.....")

#規範範圍（0-1）中的特徵向量
scaler=MinMaxScaler(feature_range=(0,1))
rescaled_features=scaler.fit_transform(global_features)
print("...Feature vector normalized...")

print("...Target labels shape: {}".format(target.shape))

#使用HDF5保存特徵向量
h5f_data=h5py.File('Output/data.h5','w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label=h5py.File('Output/label.h5','w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("...End of Vectorisation...")


# In[ ]:




