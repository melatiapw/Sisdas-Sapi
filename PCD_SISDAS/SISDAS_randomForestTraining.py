#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import
import h5py
import numpy as np
import os
import glob
import cv2
import pickle
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import metrics as ms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from SISDAS_createHistogram import *
import matplotlib.pyplot as plt
import itertools

# variables to hold the results and names
results = []
names = []
scoring = "accuracy"

#confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# import the feature vector and data labels
h5f_data = h5py.File('output/data_all.h5', 'r')
h5f_label = h5py.File('output/labels_all.h5', 'r')

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
                                                                                          test_size=0.3,
                                                                                          random_state=9)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

#Random Forest
# create the model - Random Forests
clf = RandomForestClassifier(n_estimators=100, random_state=9, verbose=2, oob_score=True)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# save the model to disk
filename = 'model_all.sav'
pickle.dump(clf, open(filename, 'wb'))

#predict test
modelrf = pickle.load(open("model_all.sav", 'rb'))
test_predict = modelrf.predict(testDataGlobal)

# Compute confusion matrix
cnf_matrix = confusion_matrix(testLabelsGlobal, test_predict)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data_labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=data_labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# 10-fold cross validation
kfold = KFold(n_splits=10, random_state=7)
#Mengetahui akurasi 10 Cross Fold Validation
cv_results = cross_val_score(modelrf, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
results.append(cv_results)
msg = "%s: %f (%f)" % ("Cross Validation Score", cv_results.mean(), cv_results.std())

#Bandingkan Akurasi hasil split data train dan data test dengan 10 Cross Validation
print ("Test Split Accuracy:", ms.accuracy_score(testLabelsGlobal,test_predict))
print(msg)

