import PIL
import os
import cv2
import numpy as np
from PIL import Image

original_directory ="C:/Users/FYP/Pictures/train/"
new_directory = "C:/Users/FYP/Pictures/train_new/"

size = 300 #rescaling size
def scaleRadius(img,size):
    shape = img.shape[0]/2
    x = img[int(shape)].sum(1)
    radius=(x>x.mean()/10).sum()/2
    final=size*1.0/radius
    return cv2.resize(img,(0,0),fx=final,fy=final)

def preprocessing(img):
    img = scaleRadius(img,size)
    colour = cv2.addWeighted(img, 4, cv2.GaussianBlur(img,(0,0), size/30), -4, 128)
    colour = cv2.resize(colour, (size,size))
    return colour

for file in os.listdir(original_directory):
    f_img = original_directory+"/"+file
    img = cv2.imread(f_img)
    img = preprocessing(img)
    print("Opening file " + file)
    cv2.imwrite(os.path.join(new_directory, file), img)

print("Preprocessing Complete")