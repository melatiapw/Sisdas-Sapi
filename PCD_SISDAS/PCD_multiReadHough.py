import os
import cv2
import numpy
import glob
import shutil

iterate_name = 1
iterate_class = 1

output_path1 = "D:/DataSapi-Train/output1"
output_path2 = "D:/DataSapi-Train/output2"
output_path3 = "D:/DataSapi-Train/output3"
output_path4 = "D:/DataSapi-Train/output4"
output_path5 = "D:/DataSapi-Train/output5"
output_path6 = "D:/DataSapi-Train/output6"
output_path7 = "D:/DataSapi-Train/output7"
output_path8 = "D:/DataSapi-Train/output8"

#scan gambar di folder masing-masing kelas yang memiliki format .bmp
for sapi_class in glob.glob("D:/DataSapi-Train/Class_1"): #untuk scan Class1
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_2"): #untuk scan Class2
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_3"): #untuk scan Class3
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_4"): #untuk scan Class4
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_5"): #untuk scan Class5
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_6"): #untuk scan Class6
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_7"): #untuk scan Class7
# for sapi_class in glob.glob("D:/DataSapi-Train/Class_8"): #untuk scan Class8
    for filename in glob.glob(os.path.join(sapi_class, "*.bmp")):
        try:
            #Mulai Hough Transform
            img_gray = cv2.imread(filename,0)
            img_gray = cv2.medianBlur(img_gray, 5)
            img_biner = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            #parameter class 1-7
            circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=350, param2=55, minRadius=0, maxRadius=0)
            # Jika circle tidak terdeteksi ubah param1 dan param2
            if(str(circles) == "None"):
                circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=320, param2=37, minRadius=0, maxRadius=0)
                # Jika circle masih tidak terdeteksi kembalikan file kosong
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
            #Substraksi gambar deteksi hough dengan gambar grayscale
            img_hasil = cv2.subtract(graykanvas, img_gray)

            namafile = filename.split("\\")[-1]
            #crop gambar hasil hough transform seukuran 112x112 piksel
            hasil_crop = img_hasil[x:x + 112, y - 56:y + 56]  # im awe [y,x]
            #Tulis hasil hough transform ke dalam folder output sesuai Class
            cv2.imwrite(os.path.join(output_path1,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_1
            # cv2.imwrite(os.path.join(output_path2,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_2
            # cv2.imwrite(os.path.join(output_path3,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_3
            # cv2.imwrite(os.path.join(output_path4,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_4
            # cv2.imwrite(os.path.join(output_path5,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_5
            # cv2.imwrite(os.path.join(output_path6,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_6
            # cv2.imwrite(os.path.join(output_path7,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_7
            # cv2.imwrite(os.path.join(output_path8,str(iterate_name) + '.jpg'), hasil_crop) #output hough transform untuk Class_8
            iterate_name+=1
            cv2.waitKey()

        except OSError as e:
            print("Something happened:", e)
