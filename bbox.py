import numpy as np
import cv2 
import os

path = "./datasets/jaad/rgb-images/images/"
list_dir = os.listdir("./datasets/jaad/rgb-images/images/")
for video in sorted(list_dir)[251:]:
    video = 
    vdo_path = path+video+"/"
    img_list = os.listdir(vdo_path)
    for images in img_list:
        # img = cv2.imread("image_path")
        img_num = images.split('.')[0]
        try :
            img = cv2.imread(vdo_path+images)
            prediction = np.loadtxt('./detections_4/images/{}/{}.txt'.format(video,img_num), delimiter = " ")
            groundTruth = np.loadtxt('./datasets/jaad/labels/images/{}/0{}.txt'.format(video,img_num), delimiter = " ")
            for gt in groundTruth:
                gt = gt.astype(int)
                x,y,x2,y2, = gt[1],gt[2],gt[3],gt[4]
                cv2.rectangle(img,(x,y),(x2,y2),(255,0,),2)
            for dt in prediction:
                dt = dt.astype(int)
                x,y,x2,y2, = dt[2],dt[3],dt[4],dt[5]
                cv2.rectangle(img,(x,y),(x2,y2),(255,0,),2)
            
            cv2.imwrite(vdo_path+'predict_'+images, img)   
            print('Done predict ',vdo_path,images)
        except:
            continue
