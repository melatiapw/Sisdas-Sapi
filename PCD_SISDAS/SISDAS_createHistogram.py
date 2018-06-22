#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import glob
import h5py
import matplotlib.pyplot as plt

# fixed-sizes for image
fixed_size = tuple((112, 112))

# path to training data
data_path = "Hough/train"

# no.of.trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.30

# seed for reproducing same results
seed = 9

# feature-descriptor: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# get the data labels
data_labels = os.listdir(data_path)

# sort the training labels
data_labels.sort()
print(data_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# loop over the training data sub-folders
for data_name in data_labels:
    # join the training data path and each species training folder
    dir = os.path.join(data_path, data_name)

    # get the current training label
    current_label = data_name

    k = 1
    # loop over the images in each sub-folder
    for file in glob.glob(dir + "/*.jpg"):

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        # image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

print ("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print ("[STATUS] data Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] data labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

print ("[STATUS] target labels: {}".format(target))
print ("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('output/data_all.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels_all.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))
h5f_data.close()
h5f_label.close()

print ("[STATUS] end of creating histogram..")
