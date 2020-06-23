# USAGE
# python basic_cnn.py

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from PIL import Image
import cv2
from imutils import paths
#import imutils
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", type=str, default="3scenes",
# 	help="path to directory containing the '3scenes' dataset")
# args = vars(ap.parse_args())

# grab all image paths in the input dataset directory, then initialize
# our list of images and corresponding class labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images("E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data"))
os.chdir("E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data")
folder_names=os.listdir()
os.chdir("E:\\PS1 SMARTi electronics\\Programs and Data")
data = []
labels = []

scale=1/20
original_shape=cv2.imread(imagePaths[0]).shape
print("original shape=",original_shape)
resized_shape=tuple([int(original_shape[0]*scale),int(original_shape[1]*scale),original_shape[2]])
print("resized_shape=",resized_shape)

shape_for_cv=tuple([resized_shape[1], resized_shape[0]])


#list_imagePaths=list(imagePaths)
# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, resize it to 32x32 pixels, scale
	# the pixel intensities to the range [0, 1], and then update our
	# images list
	image = cv2.imread(imagePath)
	# image.show()
	image = np.array(cv2.resize(image,shape_for_cv)) / 255.0
	data.append(image)

	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# encode the labels, converting them from strings to integers
lb = LabelBinarizer()
labels = lb.fit_transform(labels)



# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
	np.array(labels), test_size=0.25)

# define our Convolutional Neural Network architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(resized_shape[0],resized_shape[1], 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(20))
model.add(Activation("tanh"))
model.add(Dense(10))
model.add(Activation("tanh"))
model.add(Dense(len(folder_names)))
model.add(Activation("softmax"))


# train the model using the Adam optimizer
print("[INFO] training network...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=30, batch_size=32)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))


test=True
if test==True:
    print("testing")
    #New test_images(different date)
    testImagePaths=list(paths.list_images("E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\2020-05-21"))
    # refImagePaths=list(paths.list_images("E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Reference Images\\411010"))
    print("Number of images in dataset=", len(testImagePaths))

    #print("ref_imagepaths",refImagePaths)

    #Make folders
    for folder_name in folder_names:
        print("folder name",folder_name)
        folder=os.path.join("CNN_Categorisation_test",folder_name)
        os.makedirs(folder)

    #Make a data list
    test_data=[]
    image_no=0
    #No. of rejected image pairs
    rejects=0
    rejected_images=[]

    ##Getting data in proper format
    for testimagePath in testImagePaths:
        print("On image no.",image_no)
        image_no+=1
        image = cv2.imread(testimagePath)
        image = np.array(cv2.resize(image,shape_for_cv)) / 255.0
        image=np.expand_dims(image,axis=0)
        dissimilarities=0
        max_prediction=0
        data_to_feed=[]
        predictions=[]
        prediction=model.predict(image)[0]

        predictions=np.array(prediction)
        print("prediction is",predictions)
        prediction=predictions.flatten()
        max_prediction_ind=np.argmax(predictions)
        #print("max_prediction_ind is",max_prediction_ind)
        max_prediction=predictions[max_prediction_ind]
        #print("max_prediction is",max_prediction)
        predictions.sort()
        #print("Highest is",predictions[-1])
        #print("Second highest is",predictions[-2])
        ratio_highest_secondhighest=predictions[-1]/predictions[-2]
        #print("ratio is",ratio_highest_secondhighest)
        if max_prediction>0.5 and ratio_highest_secondhighest>1:
            folder_name=folder_names[max_prediction_ind]

            #Put image in proper location if it is similar
            folder_path=os.path.join("CNN_Categorisation_test",folder_name)
            image_put = cv2.resize(cv2.imread(testimagePath),shape_for_cv)
            # cv2.imshow("Test", image_put)
            os.chdir(folder_path)
            cv2.imwrite(testimagePath.split(os.path.sep)[-1],image_put)
            os.chdir("E:\\PS1 SMARTi electronics\\Programs and Data")
        else:
            rejects += 1
            rejected_images.append(testimagePath)
            print("dissimilar to all", testimagePath)



    print("No. of images in dataset=",len(testImagePaths))
    print("No.of rejects=", rejects)
    print("Rejected image paths")
    for rejected_image in rejected_images:
        print(rejected_image)

    for i,rejected_image in enumerate(rejected_images):
        image=cv2.imread(rejected_image)
        image=cv2.resize(image,None,fx=1/5,fy=1/5)
        cv2.imshow("Rejected image_"+str(i),image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
