import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from xml.dom import minidom
import numpy as np

#BLUE = (255,0,0)
file = 'picture.jpg'
img1 = cv2.imread(file)
label = open('./labels/video_0001/000000.txt')
#plt.imshow(img1),plt.title('ORIGINAL')
#plt.show()
a = 1
for line in label:
    plt.figure(a)
    a += 1
    temp = line.split()
    color = (int(temp[0]),0,0)
    print(int(float(temp[1])),int(float(temp[2])),int(float(temp[3])),int(float(temp[4])))
    constant = cv2.rectangle(img1,(int(float(temp[1])),int(float(temp[2]))),(int(float(temp[3])),int(float(temp[4]))),color,2)#,cv2.BORDER_CONSTANT,value=BLUE)

plt.imshow(constant)
plt.show()