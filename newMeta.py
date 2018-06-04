# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
import shutil

from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from dataExtract import *


#tempat naro file yang diupload sama user
for sapi_class in glob.glob("D:/DataSapi-Train/bisabisa"):
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
            circles = np.uint16(np.around(circles))

            # cv2.imshow("hasil", circles)
            for i in circles[0, :]:
                cv2.circle(img_biner, (i[0], i[1]), i[2], (0, 255, 255), 2)
                cv2.circle(img_biner, (i[0], i[1]), 2, (0, 0, 255), 112)

            flag = 1
            row, col, ch = img_biner.shape
            graykanvas = np.zeros((row, col, 1), np.uint8)
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
            cv2.imwrite(os.path.join(sapi_class,namafile+"crop.jpg"), hasil_crop)
            cv2.waitKey()


        except OSError as e:
            print("Something happened:", e)



#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------
# create all the machine learning models
models = []
models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))
models.append(('SVM', SVC(random_state=9)))

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

# import the feature vector and trained labels
h5f_data = h5py.File('output/data.h5', 'r')
h5f_label = h5py.File('output/labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print ("[STATUS] features shape: {}".format(global_features.shape))
print ("[STATUS] labels shape: {}".format(global_labels.shape))

print ("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=0.1,
                                                                                          random_state=9)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=100, random_state=9)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# path to test data
test_path = "Canny/test"

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_hu_moments])

    # predict label of test image
    prediction = clf.predict(global_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
