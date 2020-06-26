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
import keras



class TempleNNTrainer():
    def __init__(self):
        '''
        Creates an object of the TempleNNTrainer. Calls get_config to get info and set class variables.
        Then, based on the class attributes, will perform other functions. Option to perform training
        implicitly(when object created) or explicitly(when specific function called)
        Class attributes
        self.model:             Keras model architecture. This model will be trained by the class

        self.hyperparameters:   A dictionary containing all the hyperparameters for training
                                default ones will be used if not provided by config file

        self.output_classes:    No. of output classes((similar/dissimilar) or categories)

        self.output_labels:     Dictionary of output and its associated label

        self.model_type:        Can be useful if different kinds of models are being used.
                                eg. Similarity/Dissimilarity : Output layer can be between 0,1
                                    Categorisation  : Outputs are be softmax outputs for respective categories

        self.config_file_path:  Should be given when the object is initialised. Error logged if exception occurs

        self.log_file_path:     Path for a log file, to log the status of the training, and exceptions/errors if any

        self.save_to_path:      Path where the trained NN must be stored. Required for the save_model method

        self.previously_trained_model:  Boolean value whether a previously trained model is available or not
                                        Will be set in check_for_trained_model

        self.train_new_model:   If set to True, train a new model irrespective of whether an older model exists

        self.neural_network_id: A unique neural_network_id given to each neural network. The same temple could have
                                multiple neural networks for different purposes/categorisations.

        !!More attributes may be added with time

        '''
        pass

    def check_for_trained_model(self):
        '''
        This method checks if there is a model already available for the required temple.
        Provisions can be made to retrain a model(later), so if a model already exists,
        load the model as the class attribute
        :return:
        '''
        pass

    def get_config(self,config_file_path):
        '''
        This method takes in a  path to a config/json file as input.
        The file contains structured information on the new model to train.
        Information includes:
        Location of training data in the database.
        Hyperparameters for the NN model (like learning rate, decay, etc)
        Log file path in which information/errors during training are logged
        Path where the trained model should be stored in the database.

        INPUT: Path to config/json file
        OUTPUT: None (The relevant class attributes will be set)
        '''
        pass

    def preprocess_image(self,image):
        '''
        Return the preprocessed image. Scaling by scale factor
        :param image:
        :return:
        '''
        #First check if the resized_image_shape attribute is set. If not, set it using the scale
        if not self.resized_image_shape:
            length=image.shape[0]*self.image_scale_factor
            width=image.shape[1]*self.image_scale_factor
            new_shape=tuple([length,width,3])
            self.resized_image_shape=new_shape

        #Using scale to scale (down) image and get values from 0 - 255 to 0 - 1
        return(np.array(cv2.resize(image,tuple([self.resized_image_shape[1],self.resized_image_shape[0]])))/255.0)

    def get_training_data(self):
        '''
        This method will use the path for the training data (stored as a class attribute)
        and load data from the database into memory. Data maybe scaled down in order to save memory.
        Training data  will be prepared properly so that it can be directly used to train the NN.
        Output will be a dictionary of 2 Numpy ndarrays.
        {X:ndarray(input),Y:ndarray(target variable)}

        INPUT: None
        OUTPUT: Dictionary containing 2 Numpy ndarrays containing data from multiple images
        '''
        current_working_directory=os.getcwd()
        imagePaths = list(paths.list_images(self.training_data_path))
        os.chdir(self.training_data_path)
        self.classes = os.listdir()
        os.chdir(current_working_directory)

        data=[]
        labels=[]
        #Traverse all the paths and load the preprocessed images
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = self.preprocess_image(image)
            data.append(image)

            # extract the class label from the file path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        # encode the labels, converting them from strings to integers
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        #When no. of classes is 2, then Label Binariser doesnt behave as we want it to
        #Thus this will get it into a format we want
        if len(self.classes==2):
            labels = np.hstack((labels, 1 - labels))

        #Form the training dataset using the data and labels
        self.training_data["data"]=data
        self.training_data["labels"]=labels

        pass

    def create_model_architecture(self):
        '''
        This method will create a keras NN model architecture.
        Some parameters such as dropout, input_shape and output_shape will be given by the config file
        Some of these parameters could also have default values if they are not specified in the config file.

        INPUT: None (Inputs taken from class attributes)
        OUTPUT: None (A class attribute called model will be set by the method)
        '''
        
        # define our Convolutional Neural Network architecture
        self.model = Sequential()
        self.model.add(Conv2D(8, (3, 3), padding="same", input_shape=self.resized_image_shape))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(16, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(32, (3, 3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(20))
        self.model.add(Activation("tanh"))
        self.model.add(Dense(10))
        self.model.add(Activation("tanh"))
        self.model.add(Dense(len(self.classes)))
        self.model.add(Activation("softmax"))
        pass

    def train_model(self):
        '''
        This method uses the model architecture made before (use self.model) and trains it on the training data
        All the training logs will be written to the log file. (eg. accuracy after each epoch)
        Parameters from the config file will be used

        INPUT: None
        OUTPUT: Trained model for the particular temple
        '''
        # Assuming self.training_data is a dictionary containing data(X) and target variable(one hot encoded)

        # perform a training and testing split, using 75% of the data for
        # training and 25% for evaluation
        data=self.training_data["data"]
        labels=self.training_data["labels"]
        (trainX, testX, trainY, testY) = train_test_split(np.array(data),np.array(labels), test_size=0.25)

        # train the model using the Adam optimizer. Adding early stopping callback
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        print("[INFO] training network...")
        opt = Adam(lr=self.adam_learning_rate, decay=self.adam_decay)#Adam(lr=1e-3, decay=1e-3 / 50)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
        H = self.model.fit(trainX, trainY, validation_data=(testX, testY),epochs=30, batch_size=32,callbacks=[es_callback],verbose=0)
        pass

    def save_model(self,save_to_path):
        '''
        This method saves the trained model to the path specified in the parameter save_to_path
        This parameter should be specified by the config file.
        2 files created - a .h5 file to save the model architecture
                          a json file to save the weights
        Additional files for more info could also be stored

        :param save_to_path:
        OUPTUT: None (The model is saved to the specified path(in database or locally))
        '''
        #Make the necessary directories in the path if it doesnt exist
        if not os.path.isdir(save_to_path):
            os.makedirs(save_to_path)

        model_arch_path=os.path.join(save_to_path,"model_architecture.json")
        model_weights_path=os.path.join(save_to_path, "model_weights.h5")

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(model_arch_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(model_weights_path)

        #Log the saving of the path
        pass

    def update_database(self):
        '''
        This method can update the database table that contains the temple id and path_to the model

        INPUT: Necessary parameters to access the database
        '''
        pass



class TempleImagesPredictor():
    def __init__(self):
        '''
        This class handles the prediction of categories (or similar/dissimilar) for query images.
        It is assumed that the queries for which images need to be processed will be kept in a database.
        Alternatively, single requests from the Core Image Handler could also be taken(more info needed)
        Some attributes are:
        self.log_file_path:     Path of log file. Logs about images processed, images failed to process
                                etc. can be printed in this file.

        self.models:            A dictionary with the key as the nn_id and the value as the model

        (TO ADD)
        !!Any information that will be required to get info from the database will also be present

        '''
        pass

    def predict(self,input,model,labels):
        '''
        This method uses the model provided and the labels (in a specific format) to transform the input to an output
        Optionally, the prediction scores could also be outputted.
        Based on the kind of model used, different preprocessing to the image has to be done

        :param input: Input image/data. The image will be scaled down and transformed according to the extra info
                    given along with the model
        :param model: Trained model used for prediction
        :param labels: Label to attach to the prediction (for human-readability)
        :return: give appropriate label to the image/data, with its prediction score
        '''
        pass

    def check_database_for_queries(self):
        '''
        This method will continuously check a database for images that need to be processed.
        Once it finds images that need to be processed, it takes extra information, like the temple id
        All auxiliary information helps determine which neural network to use,etc.
        Generate a queries list with all the relevant information(preferably sorted with the NN id to be used)

        :return: A list of queries with the relevant information
        '''
        pass

    def traverse_queries(self,queries):
        '''
        This method goes through all the queries one by one. It performs the prediction process for all query images
        Sub Tasks:
        It ensures the relevant image is got from the database
        It ensures that the proper model is present in the memory to do the prediction

        :param queries: A list of query images whose output is to be predicted
        :return: queries_done: A list of queries that have been processed successfully
        '''
        pass

    def get_query_image(self):
        '''
        To get the query image from the database. Any parameters required for database operations will
        be included later.
        :return: image
        '''
        pass

    def preprocess_image(self,image):
        '''
        This method preprocesses the image before it is sent to the predict function
        eg. Turn to grayscale image, scale down th image, add dimensions to the image etc.
        :param image: image to be preprocessed
        :return: processed_image: preprocessed version of image
        '''
        pass

    def update_database_queries(self,queries_to_update):
        '''
        This method will edit the query records (corresponding to queries that have been processed)
        to state that they have been processed. Then the next run of check_database_for_queries will
        not consider those queries.

        :param queries_to_update: List of queries to mark as "processed" in the database
        :return:
        '''
        pass

    def ensure_model_present(self,nn_id):
        '''
        Given the neural network id, this method ensures that it is present in the memory. This can then be used
        for predictions. If the model is not present, then it will get the model from the database, or return an error
        (if the model is not trained/if it cannot be loaded for some reason)

        :param nn_id: Neural Network id of the required neural_network
        :return: True/False along with error code if necessary
        '''
        pass
