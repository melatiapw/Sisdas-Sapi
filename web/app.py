# from flask import Flask, render_template
from flask import Flask, request, redirect, url_for,flash,render_template,send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
# import cv2.cv as cv
import glob
import shutil
import h5py
import pickle


UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            lokasi_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(lokasi_file)
            # utama(filename)
            filenamee = filename

     #PCD
     # path to input data
    input_path = UPLOAD_FOLDER
    hasil_path = "hasilKlasifikasi"
    train_path = "Hough/train"

    # fixed-sizes for image
    fixed_size = tuple((112, 112))

    # bins for histogram
    bins = 8

    # get the training labels
    train_labels = os.listdir(train_path)

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

    iterate_name = 1
    # loop through the input images
    for file in glob.glob(input_path + "/*.bmp"):
        try:
            #disini hough
            img_gray = cv2.imread(file,0)
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

            namafile = file.split("\\")[-1]

            hasil_crop = img_hasil[x:x + 112, y - 56:y + 56]  # im awe [y,x]
            cv2.imwrite(os.path.join(hasil_path,str(iterate_name) + '.jpg'), hasil_crop)
            iterate_name+=1
            cv2.waitKey()

        except OSError as e:
            print("Something happened:", e)

    for file in glob.glob(hasil_path + "/*.jpg"):
        try:
            # baca gambar hasil Hough Transform
            image = cv2.imread(file)

            # resize gambarnya
            image = cv2.resize(image, fixed_size)

            fv_histogram  = fd_histogram(image)
            global_feature = np.hstack([fv_histogram])

            # predict label of test image
            modelrf = pickle.load(open("model.sav", 'rb'))
            prediction = modelrf.predict(global_feature.reshape(1,-1))[0]
            
            print("kelas:",train_labels[prediction])
            # show predicted label on image
            cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            namafile = file.split("\\")[-1]
            cv2.imwrite(os.path.join(input_path, namafile+"_hasil"+".jpg"), image)

        except OSError as e:
            print("Something happened:", e)
   

    return render_template('layout.html')
        
if __name__ == '__main__':
    app.run(debug=True)