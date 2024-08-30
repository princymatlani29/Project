import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import cv2
from scipy.ndimage import prewitt
from skimage import filters
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def sobeling(img):
    sobel_64 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    return sobel_64
    
def temporal(img):
    abs_64 = np.absolute(img)
    temp = np.uint8(abs_64)
    temp[temp > 70] = 255
    temp[temp <= 70] = 0
    temp = np.std(temp,axis=2)
    return temp
    
def vedio_read(path):
    vedio_feat=[]
    currentframe = 0
    cam = cv2.VideoCapture(path)
    while(True):
        ret,frame = cam.read()
        currentframe += 1
        if ret == False:
            break
        resized_frame=cv2.resize(frame, (200,200))
        sobeled = sobeling(resized_frame)
        temp = temporal(sobeled)
        vedio_feat.append(temp)
    return vedio_feat

def extract_image(vedio_ft):
    ran = int(len(vedio_ft)/10)
    img_s=[]
    spactial_temporal =[]
    for i in range(ran):
        st=10*i;lt=st+10
        v=vedio_ft[st:lt]
        spectemp = np.mean(v,axis=0)
        img_s.append(spectemp)
        cv2.imshow("spac_temp",spectemp);
        cv2.waitKey(500)
        spectemp = np.reshape(spectemp,(200,200,1))
        spactial_temporal.append(spectemp)
        if i==ran:
            v=vedio_ft[lt:len(vedio_ft)]
            spectemp = np.mean(v,axis=0)
            img_s.append(spectemp)
            cv2.imshow("spac_temp",spectemp);
            cv2.waitKey(500)
            spectemp = np.reshape(spectemp,(200,200,1))
            spactial_temporal.append(spectemp)
            
    return spactial_temporal,img_s

def img_save(path,vedio_feat):
    path1="frames_save/"+path
    alpha=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
           's','t','u','v','w','x','y','z']
    st=0;nt=0
    ff = [25*i for i in range(1,25)]
    for i in range(len(vedio_feat)):
        d = vedio_feat[i]
        cv2.imwrite(path1+alpha[st]+alpha[nt]+".png", d)
        nt+=1
        if i in ff:
            st=st+1;nt=0

path = 'Dataset/VTD.'
folder_names = os.listdir(path)
folder_names = [x for x in folder_names if x !='Thumbs.db']

label =[]
all_feature =[]
for folder in folder_names:
    path_fol = path[:-1]+"/"+folder
    ved_names = os.listdir(path_fol)
    ved_names = [x for x in ved_names if x !='Thumbs.db']
    for video in ved_names:
        path_ved = path_fol+"/"+video
        video_feature = vedio_read(path_ved)
        spactial_temp,img_s = extract_image(video_feature)
        all_feature.append(spactial_temp)
        for i in range(len(spactial_temp)):
            if "INSERTION" in path_ved:
                label.append(1)
            if "DELETION" in path_ved:
                label.append(0)
            if "DUPLICATION" in path_ved:
                label.append(2)



vedio = np.load("saved__feature/vedio_feature_vtd.npy",allow_pickle=True)
lab = np.load("saved__feature/label_vtd.npy")

vedio_feature =[]
for i in range(len(vedio)):
    vf=vedio[i]
    for j in range(len(vf)):
        vedio_feature.append(vf[j])

lb = preprocessing.LabelBinarizer()
lb.fit(lab)
label = lb.transform(lab)

vedio_feature = np.asarray(vedio_feature)
x_train,x_test,y_train,y_test = train_test_split(vedio_feature,label,test_size=0.2)


import classify
classify.main(x_train,y_train,"saved__feature/MA-EffNet_vtd")

