import os
import cv2
import numpy
import glob
import shutil



for sapi_class in glob.glob("D:\DataSapi-Train\coba"):
    for filename in glob.glob(os.path.join(sapi_class, "*.bmp")):
        try:

            #disini hough
            img_gray = cv2.imread(filename, 0)
            thresh = 127
            img_binerr = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)[1]

            img_gray = cv2.medianBlur(img_gray, 5)
            img_biner = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=225, param2=55, minRadius=0, maxRadius=0)
            #remove undetected image
            if(str(circles) == "None"):
                continue
            #     os.remove(image_path)
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

            # canny
            img_canny = cv2.Canny(hasil_crop, 100, 240)
            cv2.imwrite(os.path.join(sapi_class,namafile+"canny.jpg"), img_canny)

            #imgToCSV
            numpy.savetxt((os.path.join(sapi_class,namafile+".csv")), img_canny, delimiter=";")

            cv2.waitKey()



        except OSError as e:
            print("Something happened:", e)
