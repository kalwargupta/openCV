#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:56:15 2019

@author: jeetu
"""

import numpy as np
from PIL import Image
import os,cv2


def train_classifier(data_dir):
    #taking all the images into the list
    path=[os.path.join(data_dir,f) for f in os.listdir(data_dir)] 
    faces=[]
    ids=[]
    
    for image in path:
        img=Image.open(image).convert("L")  #converting to gray scale through PILLOW
        imageNp=np.array(img,"uint8")#image to array
        id=int(os.path.split(image)[1].split(".")[1])
        
        
        #appending the image and ids into respective list
        faces.append(imageNp)
        ids.append(id)
        
        
    # converting ids into numpy array list    
    ids=np.array(ids)
    
    clf=cv2.face.LBPHFaceRecognizer_create() 
    clf.train(faces,ids)
    clf.write("classifier.yml")
    
#calling funtion passing images in data folder
train_classifier("data")