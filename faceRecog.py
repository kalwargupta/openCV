#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:45:20 2019

@author: jeetu
"""


import cv2

def generate_dataset(img, id,img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img, classifier, scaleFactor, miniNeighbors, color, text,clf):
    gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor, miniNeighbors)
    coords=[]
    for (x, y, w, h) in features:
        cv2.rectangle(img,(x,y), (x+w,y+h), color,2)
        id,_=clf.predict(gray_img[y:y+h, x:x+w])
        if id==1:
            cv2.putText(img, "jeetu",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,cv2.LINE_AA)
        coords=[x, y, w, h]
    
    return coords

def recognize(img,clf, facfaceCascade):
    color={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"pink":(0,123,234)}
    coords=draw_boundary(img,faceCascade, 1.1, 10, color['blue'],"face",clf)
    return img
    
def detect(img, faceCascade, eyesCascade,mouthCascade, noseCascade,img_id):
    color={"blue":(255,0,0),"green":(0,255,0),"red":(0,0,255),"pink":(0,123,234)}
    coords=draw_boundary(img,faceCascade, 1.1, 10, color['blue'],"face")
    if len(coords)==4:
        roi_img=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        
        user_id=1
        generate_dataset(roi_img,user_id,img_id)
#        coords=draw_boundary(roi_img,eyesCascade, 1.1, 14, color['red'],"eye")
#        coords=draw_boundary(roi_img,noseCascade, 1.1, 18, color['pink'],"nose")
#        coords=draw_boundary(roi_img,mouthCascade, 1.1, 40, color['green'],"mouth")

    return img

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
mouthCascade=cv2.CascadeClassifier("Mouth.xml")
noseCascade=cv2.CascadeClassifier("Nariz.xml")

clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")
video_capture=cv2.VideoCapture(0)
img_id=0

while True:
    _, img=video_capture.read()
#    img=detect(img, faceCascade,eyesCascade,mouthCascade,noseCascade,img_id)
    img=recognize(img,clf,faceCascade)
    cv2.imshow("livecam", img)
    img_id+=1
    if cv2.waitKey(1) & 0xFF== ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows() 