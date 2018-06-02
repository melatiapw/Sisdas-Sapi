import os
import glob

i = 1
for sapi_class in glob.glob("D:/DataSapi-Train/Class_8-canny"):
    for filename in glob.glob(os.path.join(sapi_class, "*.jpg")):
        os.rename(filename, os.path.join(sapi_class, str(i) + '.png'))
        i += 1
