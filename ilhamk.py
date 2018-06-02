import cv2
import glob
import numpy as np
import pywt
import os
from random import randrange
from random import seed
from math import sqrt


fruit_arr = []
for fruit_dir_path in glob.glob("/home/barrillo/code/projek-pcd/coba-read-file/*"):
    if(fruit_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/1') : fruit_label = 1
    if(fruit_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/2') : fruit_label = 2
    if(fruit_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/3') : fruit_label = 3
    if(fruit_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/4') : fruit_label = 4
    if(fruit_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/5') : fruit_label = 5
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        bmean = 0
        gmean = 0
        rmean = 0

        image = cv2.imread(image_path)
        row,col,ch = image.shape
        for i in range(0,row) :
            for j in range(0,col) :
                b,g,r = image[i,j]
                bmean = bmean + int(b)
                gmean = gmean + int(g)
                rmean = rmean + int(r)

        bmean = bmean / (row*col)
        gmean = gmean / (row*col)
        rmean = rmean / (row*col)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = np.array(gray, dtype=np.float64)
        cA, (cH, cV, cD) = pywt.dwt2(data,'db1')
        Energy = (cH**2 + cV**2 + cD**2).sum()/image.size

        fruit_arr.append([bmean,gmean,rmean,Energy,fruit_label])
        fruit_np = np.array(fruit_arr)
        np.savetxt('apel.csv',fruit_np,delimiter=',')

#print fruit_arr
# with open('apel.csv', 'wb') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(fruit_arr)


##jarak euclide
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

##ngecek yg terbaik
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

#bikin random codebook
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

#iterasi
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		sum_error = 0.0
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				sum_error += error**2
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
	return codebooks

# Test the training function
seed(1)
dataset = fruit_arr
learn_rate = 0.3
n_epochs = 10
n_codebooks = 5
codebooks = train_codebooks(dataset, n_codebooks, learn_rate, n_epochs)
print('Codebooks: %s' % codebooks)

# imgtest = cv2.imread('apel1.jpg')
# graytest = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
# datatest = np.array(graytest, dtype=np.float64)
# cAtest, (cHtest, cVtest, cDtest) = pywt.dwt2(datatest,'db1')
# Energytest = (cHtest**2 + cVtest**2 + cDtest**2).sum()/imgtest.size

# hasil = get_best_matching_unit(codebooks,[Energytest,1])
# print hasil

# sapi_arr = []
# for sapi_dir_path in glob.glob("/home/barrillo/code/projek-pcd/coba-read-file/*"):
#     if(sapi_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/1') : sapi_class = 1
#     if(sapi_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/2') : sapi_class = 2
#     if(sapi_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/3') : sapi_class = 3
#     if(sapi_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/4') : sapi_class = 4
#     if(sapi_dir_path.split("\\")[-1] == '/home/barrillo/code/projek-pcd/coba-read-file/5') : sapi_class = 5

#im_gray = cv2.imread("sapi2.bmp",0)
