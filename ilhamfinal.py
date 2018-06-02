import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import os

for sapi_dir_path in glob.glob("D:/Data Sapi/data/dataTRAIN/Class_7"):
    for image_path in glob.glob(os.path.join(sapi_dir_path, "*.bmp")):
        im_gray = cv2.imread(image_path, 0)
        im_gray = cv2.medianBlur(im_gray,5)
        im_biner = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(im_gray, cv2.HOUGH_GRADIENT,1,20,param1=550, param2=36,
                           minRadius=0, maxRadius=0)
        #remove undetected image
        if(str(circles) == "None"):
            continue
        #     os.remove(image_path)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(im_biner, (i[0],i[1]), i[2], (0,255,255), 2)
            cv2.circle(im_biner, (i[0],i[1]), 2, (0,0,255), int(2.1*i[2]))
        acuan = int(2.1*i[2])
        flag=1
        row, col, ch = im_biner.shape
        graykanvas = np.zeros((row, col, 1), np.uint8)
        for i in range(0, row):
            for j in range(0, col):
                b, g, r = im_biner[i, j]
                if(b == 255 & g==0 & r==0):
                    graykanvas.itemset((i,j,0), 255)
                    if(flag==1):
                        x=i
                        y=j
                        flag = 100
                else:
                    graykanvas.itemset((i,j,0), 0)

        im_hasil = cv2.subtract(graykanvas, im_gray)

        namafile = image_path.split("\\")[-1]

        hasil_crop = im_hasil[x:x+int(acuan*1.3), y-int(acuan/1.3):y+int(acuan/1.3)] #im awe [y,x]
        cv2.imwrite(os.path.join(sapi_dir_path,namafile+"crop.jpg"), hasil_crop)
