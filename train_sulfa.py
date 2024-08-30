import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import cv2
from scipy.ndimage import prewitt
from skimage import filters
import os


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
        spectemp = np.reshape(spectemp,(200,200,1))
        spactial_temporal.append(spectemp)
        if i==ran:
            v=vedio_ft[lt:len(vedio_ft)]
            spectemp = np.mean(v,axis=0)
            img_s.append(spectemp)
            spectemp = np.reshape(spectemp,(200,200,1))
            spactial_temporal.append(spectemp)
            
    return spactial_temporal,img_s

path = 'Dataset/SULFA.'
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
        print(path_ved)
        for i in range(len(spactial_temp)):
            if "ORIGINAL" in path_ved:
                label.append(0)
            if "DELETION" in path_ved:
                label.append(1)
            if "DUPLICATION" in path_ved:
                label.append(2)
            if "INSERTION" in path_ved:
                label.append(3)
        

video = np.load("saved__feature/vedio_feature_sulfa.npy",allow_pickle=True)
lab = np.load("saved__feature/label_sulfa.npy")

video_feature =[]
for i in range(len(video)):
    vf=video[i]
    for j in range(len(vf)):
        video_feature.append(vf[j])
        
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(lab)
label = lb.transform(lab)

video_feature = np.asarray(video_feature)
x_train,x_test,y_train,y_test = train_test_split(video_feature,label,test_size=0.2)



import classify
classify.main(x_train,y_train,"saved__feature/MA-EffNet_sulfa")

