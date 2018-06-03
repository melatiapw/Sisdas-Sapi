import os
import cv2
import numpy
import glob
import shutil


iterate_nama = 1
for sapi_class in glob.glob("D:/DataSapi-Train/bisain"):
    for filename in glob.glob(os.path.join(sapi_class, "*.bmp")):
        try:
            #disini hough
            img_gray = cv2.imread(filename,0)
            img_gray = cv2.medianBlur(img_gray, 5)
            img_biner = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            #parameter class 1-7
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=350, param2=55, minRadius=0, maxRadius=0)
            if(str(circles) == "None"):
                circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=320, param2=37, minRadius=0, maxRadius=0)
                if(str(circles) == "None"):
                    continue
            circles = numpy.uint16(numpy.around(circles))

            # cv2.imshow("hasil", circles)
            for i in circles[0, :]:
                cv2.circle(img_biner, (i[0], i[1]), i[2], (0, 255, 255), 2)
                cv2.circle(img_biner, (i[0], i[1]), 2, (0, 0, 255), 112)

            flag = 1
            row, col, ch = img_biner.shape
            graykanvas = numpy.zeros((row, col, 1), numpy.uint8)
            for i in range(0, row):
                for j in range(0, col):
                    b, g, r = img_biner[i, j]
                    if (b == 255 & g == 0 & r == 0):
                        graykanvas.itemset((i, j, 0), 255)
                        if (flag == 1):
                            x = i
                            y = j
                            flag = 100
                    else:
                        graykanvas.itemset((i, j, 0), 0)

            img_hasil = cv2.subtract(graykanvas, img_gray)

            namafile = filename.split("\\")[-1]

            hasil_crop = img_hasil[x:x + 112, y - 56:y + 56]  # im awe [y,x]
            # cv2.imwrite(os.path.join(sapi_class,namafile+"crop.jpg"), hasil_crop)
            cv2.imwrite(os.path.join(sapi_class,str(iterate_nama)+".jpg"), hasil_crop)
            iterate_nama += 1

            cv2.waitKey()
            # # canny
            # img_canny = cv2.Canny(hasil_crop, 100, 240)
            # cv2.imwrite(os.path.join(sapi_class,str(iterate_nama)+".jpg"), img_canny)
            # iterate_nama += 1
            #
            # cv2.waitKey()

        except OSError as e:
            print("Something happened:", e)
