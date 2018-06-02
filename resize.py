import os
import cv2
import glob



for sapi_class in glob.glob("D:/DataSapi-Train/Class_8-full"):      #nama kelasnya diganti sesuai kelas yg mau di run
    for filename in glob.glob(os.path.join(sapi_class, "*.bmp")):
        try:
            fixed_size = tuple((150, 250))
            #image ya gambarnya
            filename = cv2.resize(filename, fixed_size)
            cv2.imwrite(os.path.join(sapi_class,namafile+"resize.jpg"), filename)
            cv2.waitKey()
