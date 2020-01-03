#!/usr/bin/env python
# coding: utf-8

# In[1]:


from feature_train import *
from feature_test import *
import pandas as pd
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:


output = []

h5f_data = h5py.File('Output/data.h5', 'r')
h5f_label = h5py.File('Output/label.h5', 'r')

h5f_test_data = h5py.File('Output/test_data.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_test_features_string = h5f_test_data['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
global_test_features = np.array(global_test_features_string)

h5f_data.close()
h5f_label.close()
h5f_test_data.close()

# 驗證特徵向量和標籤的形狀
print(".... features shape: {}".format(global_features.shape))
print(".... labels shape: {}".format(global_labels.shape))

                                                                                         
trainDataGlobal = global_features
trainLabelsGlobal = global_labels
testDataGlobal = global_test_features

print("... Splitted Train and Test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))

# 過濾所有警告
import warnings
warnings.filterwarnings('ignore')


# In[3]:


clf  = RandomForestClassifier(n_estimators=100, random_state=100,min_samples_split=100)


clf.fit(trainDataGlobal, trainLabelsGlobal)
predic_ = clf.predict(testDataGlobal)

for i in predic_:
    if(i==0):
        output.append('5498592')
    elif(i==1):
        output.append("5513628")
    elif(i==2):
        output.append("5540978")
    elif(i==3):
        output.append("5544358")
    elif(i==4):
        output.append("5549723")
    elif(i==5):
        output.append("5553911")
    elif(i==6):
        output.append("5554445")
    elif(i==7):
        output.append("5559534")
    elif(i==8):
        output.append("5559738")
    elif(i==9):
        output.append("5562809")
    elif(i==10):
        output.append("5569034")
    elif(i==11):
        output.append("5569981")
    elif(i==12):
        output.append("5583008")
    elif(i==13):
        output.append("5586718")
    elif(i==14):
        output.append("5587543")
    elif(i==15):
        output.append("5592004")
    elif(i==16):
        output.append("5597972")
    elif(i==17):
        output.append("5604363")
    elif(i==18):
        output.append("5605786")
    elif(i==19):
        output.append("5613764")
    elif(i==20):
        output.append("5617665")
    elif(i==21):
        output.append("5621307")
    elif(i==22):
        output.append("5625014")
    elif(i==23):
        output.append("5629608")
    elif(i==24):
        output.append("5630205")
    elif(i==25):
        output.append("5652218")
    elif(i==26):
        output.append("5655015")
    elif(i==27):
        output.append("5657590")
    elif(i==28):
        output.append("5664432")
    elif(i==29):
        output.append("5666714")
    elif(i==30):
        output.append("5675606")
    elif(i==31):
        output.append("5680642")
    elif(i==32):
        output.append("5681403")
    elif(i==33):
        output.append("5681882")
    elif(i==34):
        output.append("5693349")
    elif(i==35):
        output.append("5694385")
    elif(i==36):
        output.append("5702934")
    elif(i==37):
        output.append("5713413")
    elif(i==38):
        output.append("5725535")
    elif(i==39):
        output.append("5725655")
    elif(i==40):
        output.append("5728349")
    elif(i==41):
        output.append("5734180")
    elif(i==42):
        output.append("5743918")
    elif(i==43):
        output.append("5768706")
    elif(i==44):
        output.append("5786096")
    elif(i==45):
        output.append("5801165")
    elif(i==46):
        output.append("5803970")
    elif(i==47):
        output.append("5804957")
    elif(i==48):
        output.append("5809867")
    elif(i==49):
        output.append("5811827")
    elif(i==50):
        output.append("5831542")
    elif(i==51):
        output.append("5847762")
    elif(i==52):
        output.append("5861947")
    elif(i==53):
        output.append("5873163")
    elif(i==54):
        output.append("5873274")
    elif(i==55):
        output.append("5875151")
    elif(i==56):
        output.append("5902240")
    elif(i==57):
        output.append("5903584")
    elif(i==58):
        output.append("5908190")
    elif(i==59):
        output.append("5911712")
    elif(i==60):
        output.append("5928307")
    elif(i==61):
        output.append("5934349")
    elif(i==62):
        output.append("5937337")
    elif(i==63):
        output.append("5942362")
    elif(i==64):
        output.append("5949163")
    elif(i==65):
        output.append("5961035")
    elif(i==66):
        output.append("5961035")
    elif(i==67):
        output.append("5997749")
    elif(i==68):
        output.append("6002497")
    elif(i==69):
        output.append("6031709")
    elif(i==70):
        output.append("6039617")
    elif(i==71):
        output.append("6062003")
    elif(i==72):
        output.append("6074154")
    elif(i==73):
        output.append("6079503")
    elif(i==74):
        output.append("6109225")
    elif(i==75):
        output.append("6121003")
    elif(i==76):
        output.append("6152976")
    elif(i==77):
        output.append("6166434")
    elif(i==78):
        output.append("6188016")
    elif(i==79):
        output.append("6189742")
    elif(i==80):
        output.append("6212900")
    elif(i==81):
        output.append("6219361")
    elif(i==82):
        output.append("6230258")
    elif(i==83):
        output.append("6234845")
    elif(i==84):
        output.append("6249350")
    elif(i==85):
        output.append("6252408")
    elif(i==86):
        output.append("6266630")
    elif(i==87):
        output.append("6293841")
    elif(i==88):
        output.append("6304911")
    elif(i==89):
        output.append("6305385")
    elif(i==90):
        output.append("6317420")
    elif(i==91):
        output.append("6353801")
    elif(i==92):
        output.append("6383100")
    elif(i==93):
        output.append("6392685")
    elif(i==94):
        output.append("6418193")
    elif(i==95):
        output.append("6432391")
    elif(i==96):
        output.append("6487924")
    elif(i==97):
        output.append("6489352")
    elif(i==98):
        output.append("6549862")
    elif(i==99):
        output.append("6573530")

pd.DataFrame(output).to_excel('output.xlsx',header=False,index=False)


# In[ ]:




