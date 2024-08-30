from tensorflow.keras.models import load_model
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from scipy.ndimage import prewitt
from skimage import filters
import pandas as pd
import numpy as np
import os
import cv2
from tkinter import Tk    
from tkinter.filedialog import askopenfilename


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
        spactial_temporal.append(spectemp)
        if i==ran:
            v=vedio_ft[lt:len(vedio_ft)]
            spectemp = np.mean(v,axis=0)
            spactial_temporal.append(spectemp)  
    return spactial_temporal


x_test=np.load("saved__feature/x_test_sulfa.npy")
y_test=np.load("saved__feature/y_test_sulfa.npy")

model = load_model("saved__feature/MA-EffNet_sulfa")
predition = model.predict(x_test)

Tk().withdraw() 
path_file = askopenfilename() 

video_feature = vedio_read(path_file)
spactial_temp = extract_image(video_feature)
pred = Model.predict(spactial_temp,path_file)


