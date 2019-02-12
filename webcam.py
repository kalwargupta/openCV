#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:12:05 2019

@author: jeetu
"""

import cv2

def draw_boundary(img, classifier, scaleFactor, miniNeighbors, color, text):
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor, miniNeighbors)
    coords=[]
    for (x, y, w, h) in features:
        cv2.rectangle(img,(x,y), (x+w,y+h), color,2)
        cv2.putText(img, text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,cv2.LINE_AA)
        coords=[x, y, w, h]
    
    return coords

def detect(img, faceCascade, eyesCascade,mouthCascade, noseCascade):
    color={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"pink":(0,123,234)}
    coords=draw_boundary(img,faceCascade, 1.1, 10, color['blue'],"face")
    if len(coords)==4:
        roi_img=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        coords=draw_boundary(roi_img,eyesCascade, 1.1, 14, color['red'],"eye")
        coords=draw_boundary(roi_img,noseCascade, 1.1, 18, color['pink'],"nose")
        coords=draw_boundary(roi_img,mouthCascade, 1.1, 40, color['green'],"mouth")

    return img

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
mouthCascade=cv2.CascadeClassifier("Mouth.xml")
noseCascade=cv2.CascadeClassifier("Nariz.xml")


video_capture=cv2.VideoCapture(0)

while True:
    _, img=video_capture.read()
    img=detect(img, faceCascade,eyesCascade,mouthCascade,noseCascade)
    cv2.imshow("livecam", img)
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows() 