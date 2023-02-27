import cv2
import numpy as np
import os
import argparse
import random
from PIL import Image

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-i', '--input_root', type=str, required=False, default='in')
parser.add_argument('-o', '--output_path', type=str, required=False, default='out')
parser.add_argument('-a', '--asset_path', type=str, required=False, default='assets')
parser.add_argument('-m', '--mode', type=str, required=False, default='replaceF')
args = parser.parse_args()

input_path = args.input_root
output_path = args.output_path
asset_path = args.asset_path
mode = args.mode

face_cascade = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('lib/haarcascade_eye.xml')
count = 0
assets = []
if(mode == 'replace' or mode == 'replaceG' or mode == 'replaceFG' or mode == 'replaceF'):
   for filepath in os.listdir(asset_path):
    assets.append(asset_path+'/{0}'.format(filepath))


for file  in os.listdir(input_path):

    filename = os.fsdecode(file)
    img = cv2.imread(input_path+'/'+filename)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    dst = np.full((img.shape[0],img.shape[1], 3), (0, 255, 0), dtype=('uint8'))
    masked = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = img.shape[:2]

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
    if(mode == 'track'):                                                 #Track Mode
        for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 150, 0), 2)
        for (x, y, w, h) in eyes:
         cv2.rectangle(img, (x, y), (x+w, y+h), (150, 255, 0), 2)
         cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        masked = cv2.bitwise_and(img, img, img,mask=mask)
    
    if(mode == 'trackG'):                                                 #Track Mode
        for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 150, 0), 2)
        for (x, y, w, h) in eyes:
         cv2.rectangle(img, (x, y), (x+w, y+h), (150, 255, 0), 2)
         cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        masked = cv2.bitwise_and(img, img, dst,mask=mask)
    
    if(mode == 'replace'):                                              #Replace Mode
        img = Image.open(input_path+'/'+filename)                                           
        for (x, y, w, h) in eyes:
         tmp = Image.open(assets[random.randint(0,11)])
         tmp = tmp.resize((int(w), int(h)), Image.LANCZOS)
         img.paste(tmp, (int(x),int(y)))
    
    if(mode == 'replaceF'):                                              #Replace Mode
        img = Image.open(input_path+'/'+filename)                                           
        for (x, y, w, h) in eyes:
         tmp = Image.open(assets[random.randint(0,11)])
         tmp = tmp.resize((int(w), int(h)), Image.LANCZOS)
         img.paste(tmp, (int(x),int(y)))
        for(x,y,w,h) in faces:
         tmp = img.crop((x,y,(x+w),(y+h)))
         img = Image.open(input_path+'/'+filename)
         img.paste(tmp, (int(x),int(y)))

    if(mode == 'replaceFG'):                                              #Replace Mode
        img = Image.open(input_path+'/'+filename)                                           
        for (x, y, w, h) in eyes:
         tmp = Image.open(assets[random.randint(0,11)])
         tmp = tmp.resize((int(w), int(h)), Image.LANCZOS)
         img.paste(tmp, (int(x),int(y)))
        for(x,y,w,h) in faces:
         tmp = img.crop((x,y,(x+w),(y+h)))
         img = Image.new(mode='RGB', size=(height,width), color=(0,255,0))
         img.paste(tmp, (int(x),int(y)))
        
    if(mode == 'replaceG'):                                             #Replace Mode
        img = Image.new(mode='RGB', size=(height,width), color=(0,255,0))                                          
        for (x, y, w, h) in eyes:
         tmp = Image.open(assets[random.randint(0,11)])
         tmp = tmp.resize((int(w), int(h)), Image.LANCZOS)
         img.paste(tmp, (int(x),int(y)))     
         
         
        #masked = cv2.bitwise_and(test, test, dst,mask=mask)
        
    if(mode == 'track' or mode == 'trackG'):
        #cv2.imwrite(output_path+'/reslut_'+ str(count) +'.png', masked)
        cv2.imwrite(output_path+'/'+filename, masked)
    if(mode == 'replace' or mode == 'replaceG' or mode == 'replaceFG' or mode == 'replaceF'):
       #img.save(output_path+'/reslut_'+ str(count) +'.png')
       img.save(output_path+'/'+filename)
    count = count + 1
    continue

cv2.waitKey(0)
