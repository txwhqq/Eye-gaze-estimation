# -*- coding: utf-8 -*-
from utils.utils import *
import cv2

import argparse

import scipy.misc
from socket import *

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.utils.data as data
import os

import scipy.misc
import numpy as np


a1 = ()
data1 = 'w'


device = torch.device('cuda:0')

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

class FeatureExract(nn.Module):
    def __init__(self,num_classes):

        super(FeatureExract,self).__init__()
        resnet = models.resnet18(pretrained=True)   #
        modules = list(resnet.children())[:-1]      #
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features,num_classes,bias=True)  #
        
    
    ##
    def forward(self,images):
        features = self.resnet(images)
        #print(features[0,1])
        features = features.reshape(features.size(0),-1)
        features = self.linear(features)
        return features

model = torch.load('F:\\Gaze_estimator\\weights\\model_18.ckpt')  ##load your model here
model = model.to(device)

Theta = np.array([[-0.5601,9.4961,-59.1749,-57.2073],[0.0002,-0.0001,-0.0003,-0.0005],[0.0741,0.8672,0.5349,0.8576]])
face_haar = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
#cv2.namedWindow("Image")
#cv2.CAP_DSHOW
#cv2.CAP_DSHOW+
cap = cv2.VideoCapture(cv2.CAP_DSHOW+1)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
num = 0

model.eval()
with torch.no_grad():
    while (cap.isOpened()):#
        
        ret,img = cap.read()  #
        #print(ret)
        if ret == True: ##
            #cv2.imshow('Image',img)
            #cv2.waitKey(10)
            faces = face_haar.detectMultiScale(img, 1.3, 5)
            img2 = img.copy()
            print(type(faces))
            if type(faces) != type(a1):
                rect_size = faces[:,2]*faces[:,3]
                max_ind = np.argmax(rect_size)
                if max_ind.size !=0:
                    face_x = faces[max_ind,0]
                    face_y = faces[max_ind,1]
                    face_w = faces[max_ind,2]
                    face_h = faces[max_ind,3]
                    a=np.array([[1,face_w*face_w,face_w]])
                    rect = np.dot(a,Theta)
                    
                    eye_x1 = face_x+int(rect[0][0])
                    eye_x2 = face_x+int(rect[0][1])
                    eye_y1 = face_y+int(rect[0][2])
                    eye_y2 = face_y+int(rect[0][3])
                    cropImg = img[eye_y1:eye_y2,eye_x1:eye_x2]
                    cropImg = cv2.resize(cropImg,(244,52))
                    #cv2.rectangle(img,(face_x+int(rect[0][0]),face_y+int(rect[0][2])),(face_x+int(rect[0][1]),face_y+int(rect[0][3])),(0,255,0),2)
                    #cv2.imwrite('F:\\Gaze_estimator\\cam_eyedetection\\'+str(num)+'.jpg',cropImg)
                    num=num+1
                    B,G,R = cv2.split(cropImg)
                    cropImg = cv2.merge([R,G,B])
                    image1 = cropImg
                    #image1 = scipy.misc.imresize(cropImg,144,52)
                    #scipy.misc.imsave('F:\\Gaze_estimator\\cam_eyedetection\\'+str(num)+'.jpg',image1)
                    #image1 = scipy.misc.imresize(cropImg, (image_size, image_size), interp='bilinear')
                    image2 = prewhiten(image1)
                    image2 = image2.astype(np.float32)
                    image2 = image2.reshape(-1,3,52,244)
                    image2 = torch.from_numpy(image2)
                    image2 = image2.to(device)
                    
                    outputs = model(image2)
                    _,argmax = torch.max(outputs,1)
                    print(argmax+1)
                    
                    
                    
                    #img3 = image2.reshape(-1,image_size,image_size,3)
                    #print(img3.shape)
        #               print(image2.shape)
                    #data1 = argmax+1
                    #if data1 ==1:
                     #   data1 = 'a'
                    #elif data1 ==2:
                     #   data1 = 's'
                    #else:
                     #   data1 = 'd'
                    #tcpCliSock.send(data1.encode())
                    #data1 = tcpCliSock.recv(BUFSIZ)
                    
                    
                    cv2.rectangle(img2,(face_x+int(rect[0][0]),face_y+int(rect[0][2])),(face_x+int(rect[0][1]),face_y+int(rect[0][3])),(0,255,0),2)
                    img2 =cv2.resize(img2,(1920,1080),interpolation=cv2.INTER_AREA) 
                    
                    #scipy.misc.imshow('Image',image1)
                    
        #              # if k==27:
        #              #    break
                cv2.imshow('Image',img2)
                k = cv2.waitKey(100)
                if k==27:
                    break
#                    
#            

cap.release

cv2.destroyAllWindows()
