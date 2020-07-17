# import the necessary packages

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from PIL import Image
import cv2
from imutils import paths
# import imutils
import numpy as np
import argparse
import os
# import tensorflow.keras
import json
import traceback
import base64
import datetime


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

        :param config_file_path: Path of the config file, to configure the neural network
        '''
        # Initialising all attributes

        #self.config_file_path = config_file_path
        self.error_log_file_path = None
        self.status_log_file_path=None
        self.training_data_path = None
        self.testing_data_path = None
        self.save_model_path = None
        self.temple_id = None

        self.model = None
        self.hyperparameters = {}
        self.classes = None
        self.resized_image_shape = None
        self.image_scale_factor = 1 / 20

        self.training_data = {}
        self.testing_data = {}
        self.testing_accuracy = 0

        # # Calling get_config() to get config file and set up the attributes
        # self.get_config(config_file_path)
        #
        # self.error_logger("--------------------------------------------------------------------------------------------")

        pass

    def set_paths(self, temple_id, model_path, training_data_path, testing_data_path, log_path):
        # Setting paths based on temple_id
        self.save_model_path = os.path.join(model_path, str(temple_id))
        self.training_data_path = os.path.join(training_data_path, str(temple_id))
        self.testing_data_path = os.path.join(testing_data_path, str(temple_id))

        self.error_log_file_path = os.path.join(log_path, "error_log.txt")
        self.status_log_file_path = os.path.join(log_path, "log.txt")

        self.temple_id = temple_id

        self.status_logger("Paths to directories set")

    def start_training(self):
        # Now getting training data from the database in order to train the model
        self.get_training_data()

        self.status_logger("Training data has been loaded")

        # Creating the architecture of the model
        self.create_model_architecture()

        # Training the model
        self.train_model()

        self.status_logger("Model has been trained")

        # Getting testing data
        self.get_testing_data()

        self.status_logger("Testing data is loaded")

        # Testing model
        self.test_model()

        self.status_logger("Model has been tested. Testing accuracy is "+str(self.testing_accuracy))
        # Save the model in the path specified
        self.save_model(self.save_model_path)

    def check_for_trained_model(self):
        '''
        This method checks if there is a model already available for the required temple.
        Provisions can be made to retrain a model(later), so if a model already exists,
        load the model as the class attribute
        :return:
        '''
        pass

    def get_config(self, config_file_path):
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

        try:
            # Opening and reading contents of file as json
            with open(config_file_path, 'r') as config_file:
                config_json = json.load(config_file)

            # Now we have the json file. We'll set the attributes accordingly
            self.training_data_path = config_json["training data path"]
            self.testing_data_path = config_json["testing data path"]
            # self.hyperparameters=config_json["hyperparameters"]
            self.log_file_path = config_json["log file path"]
            self.save_model_path = config_json["save model path"]
            self.temple_id = config_json["temple id"]

        ##Catching all errors as tracebacks are logged
        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)

        pass

    def preprocess_image(self, image):
        '''
        Return the preprocessed image. Scaling by scale factor
        :param image:
        :return:
        '''
        try:
            # First check if the resized_image_shape attribute is set. If not, set it using the scale
            if not self.resized_image_shape:
                length = image.shape[0] * self.image_scale_factor
                width = image.shape[1] * self.image_scale_factor
                new_shape = tuple([int(length), int(width), 3])
                self.resized_image_shape = new_shape

            # Using scale to scale (down) image and get values from 0 - 255 to 0 - 1
            processed_image = (np.array(cv2.resize(image, tuple([self.resized_image_shape[1],
                                                                 self.resized_image_shape[0]]))) / 255.0)

        ##Catching all errors as tracebacks are logged
        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
            return (None)

        else:
            return (processed_image)

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
        try:
            # Getting the classes to categorise in
            current_working_directory = os.getcwd()
            imagePaths = list(paths.list_images(self.training_data_path))
            os.chdir(self.training_data_path)
            self.classes = os.listdir()
            os.chdir(current_working_directory)

            data = []
            labels = []
            # Traverse all the paths and load the preprocessed images
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
            # When no. of classes is 2, then Label Binariser doesnt behave as we want it to
            # Thus this will get it into a format we want
            if len(self.classes) == 2:
                labels = np.hstack((labels, 1 - labels))

            # Form the training dataset using the data and labels
            self.training_data["data"] = data
            self.training_data["labels"] = labels

        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)

        pass

    def create_model_architecture(self):
        '''
        This method will create a keras NN model architecture.
        Some parameters such as dropout, input_shape and output_shape will be given by the config file
        Some of these parameters could also have default values if they are not specified in the config file.

        INPUT: None (Inputs taken from class attributes)
        OUTPUT: None (A class attribute called model will be set by the method)
        '''
        try:
            dropout_prob = 0.5
            # define our Convolutional Neural Network architecture
            self.model = Sequential()
            self.model.add(Conv2D(8, (3, 3), padding="same", input_shape=self.resized_image_shape))
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Dropout(dropout_prob))
            self.model.add(Conv2D(16, (3, 3), padding="same"))
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Dropout(dropout_prob))
            # self.model.add(Conv2D(32, (3, 3), padding="same"))
            # self.model.add(Activation("relu"))
            # self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            # self.model.add(Dropout(dropout_prob))
            self.model.add(Flatten())
            self.model.add(Dense(20))
            self.model.add(Activation("tanh"))
            self.model.add(Dropout(dropout_prob))
            self.model.add(Dense(10))
            self.model.add(Activation("tanh"))
            self.model.add(Dropout(dropout_prob))
            self.model.add(Dense(len(self.classes)))
            self.model.add(Activation("softmax"))


        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
        pass

    def train_model(self):
        '''
        This method uses the model architecture made before (use self.model) and trains it on the training data
        All the training logs will be written to the log file. (eg. accuracy after each epoch)
        Parameters from the config file will be used

        INPUT: None
        OUTPUT: Trained model for the particular temple
        '''
        try:

            # Assuming self.training_data is a dictionary containing data(X) and target variable(one hot encoded)

            # perform a training and testing split, using 75% of the data for
            # training and 25% for evaluation
            data = self.training_data["data"]
            labels = self.training_data["labels"]

            # print("Training data is",data)
            # print("Training labels are",labels)
            (trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

            # train the model using the Adam optimizer. Adding early stopping callback
            # es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
            # print("[INFO] training network...")
            opt = Adam(lr=1e-3,
                       decay=1e-3 / 50)  # Adam(lr=self.hyperparameters["adam_learning_rate"], decay=self.hyperparameters["adam_decay"])
            self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
            H = self.model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100,
                               batch_size=32)  # ,callbacks=[es_callback])
            print("H is")
            print(H)


        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
        pass

    def get_testing_data(self):
        imagePaths = list(paths.list_images(self.testing_data_path))
        print("Testing image paths are", imagePaths)

        data = []
        labels = []
        # Traverse all the paths and load the preprocessed images
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = self.preprocess_image(image)
            data.append(image)

            # extract the class label from the file path and update the labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        # Form the testing dataset using the data and labels
        self.testing_data["data"] = data
        self.testing_data["labels"] = labels

        # print("Testing data is",self.testing_data["data"])
        # print("Testing labels is",self.testing_data["labels"])

    def test_model(self):
        no_of_images = len(self.testing_data["data"])
        correct_classifications = 0
        labels = self.testing_data["labels"]
        # print("Testing data dimensions=",np.array(self.testing_data["data"]))
        predictions = self.model.predict(np.array(self.testing_data["data"]))
        max_inds = np.argmax(predictions, axis=1)
        max_prob = np.amax(predictions, axis=1)

        for i in range(no_of_images):
            if max_prob[i] > 0.5 and self.classes[max_inds[i]] == labels[i]:
                correct_classifications += 1

        self.testing_accuracy = correct_classifications / no_of_images
        print("Testing accuracy is", self.testing_accuracy)

    def save_model(self, save_to_path):
        '''
        This method saves the trained model to the path specified in the parameter save_to_path
        This parameter should be specified by the config file.
        2 files created - a .h5 file to save the model architecture
                          a json file to save the weights
        Additional files for more info could also be stored

        :param save_to_path:
        OUPTUT: None (The model is saved to the specified path(in database or locally))
        '''
        try:
            #save_to_path = os.path.join(save_to_path, self.temple_id)
            # Make the necessary directories in the path if it doesnt exist
            if not os.path.isdir(save_to_path):
                os.makedirs(save_to_path)

            model_arch_path = os.path.join(save_to_path, "model_architecture.json")
            model_weights_path = os.path.join(save_to_path, "model_weights.h5")
            model_extra_info_path = os.path.join(save_to_path, "extra_info.json")

            # serialize model to JSON
            model_json = self.model.to_json()
            with open(model_arch_path, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(model_weights_path)
            # serialize extra info to json
            extra_info = {}
            extra_info["class labels"] = self.classes
            extra_info["resized image shape"] = self.resized_image_shape
            with open(model_extra_info_path, 'w') as extra_info_file:
                json.dump(extra_info, extra_info_file, indent=4)

            # Log the saving of the model
            self.status_logger("Model saved in " + save_to_path)

        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)

        pass

    def update_database(self):
        '''
        This method can update the database table that contains the temple id and path_to the model

        INPUT: Necessary parameters to access the database
        '''
        pass

    def error_logger(self, message):
        with open(self.error_log_file_path, 'a') as log_file:
            log_file.write("[Error Log at "+str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+" ]")
            log_file.write("\n")
            log_file.write(message)
            log_file.write("\n")

    def status_logger(self,message):
        with open(self.status_log_file_path, 'a') as log_file:
            log_file.write("[Status Log at "+str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+" ]")
            log_file.write("\n")
            log_file.write(message)
            log_file.write("\n")


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
        # self.query_file_path=query_file_path
        self.path_to_models = None
        self.input = None
        self.temple_id = None

        self.predicted_classes = None
        self.models = {}
        self.min_confidence = 0.5

        self.error_log_file_path = None
        self.status_log_file_path=None

        self.current_model = None
        self.resized_image_shape = None
        self.class_labels = None

        # self.parse_query_file()

        #self.error_logger("---------------------------------------------------------------------")

        pass

    def get_model(self, nn_id):
        # Path of the model to get
        get_model_path = os.path.join(self.path_to_models, str(nn_id))

        model_arch_path = os.path.join(get_model_path, "model_architecture.json")
        model_weights_path = os.path.join(get_model_path, "model_weights.h5")
        model_extra_info_path = os.path.join(get_model_path, "extra_info.json")

        # Making the model from files
        # Getting model architecture
        with open(model_arch_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        # Getting model weights
        model.load_weights(model_weights_path)

        # Getting extra info
        # Opening and reading contents of file as json
        with open(model_extra_info_path, 'r') as extra_info_file:
            extra_info_json = json.load(extra_info_file)

        # Now we have the json file. We'll set the attributes accordingly
        model_class_labels = extra_info_json["class labels"]
        model_resized_image_shape = tuple(extra_info_json["resized image shape"])

        # Compiling model to make it usable
        opt = Adam(lr=1e-3,
                   decay=1e-3 / 50)  # Adam(lr=self.hyperparameters["adam_learning_rate"], decay=self.hyperparameters["adam_decay"])
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

        # Adding the compiled model and extra info to models
        self.models[nn_id] = {}
        self.models[nn_id]["model"] = model
        self.models[nn_id]["class_labels"] = model_class_labels
        self.models[nn_id]["resized_image_shape"] = model_resized_image_shape

    def load_model(self, nn_id):
        '''
        Using the nn_id, this function loads the model (present in self.models) as the current_model.
        The other attributes such as resized_image_size, labels etc will be reset to match the
        one appropriate to the current_model
        :param nn_id: Neural Network Id
        :return:
        '''
        try:
            # If nn_id not there in models, get it
            if nn_id not in self.models:
                self.get_model(nn_id)

            # Loading the model first
            self.current_model = self.models[nn_id]["model"]

            self.resized_image_shape = self.models[nn_id]["resized_image_shape"]
            self.class_labels = self.models[nn_id]["class_labels"]
            self.temple_id = nn_id

        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)

    def preprocess_image(self, image):
        '''
        Return the preprocessed image. Scaling by scale factor
        :param image:
        :return:
        '''
        try:
            # Using scale to scale (down) image and get values from 0 - 255 to 0 - 1
            processed_image = (np.array(cv2.resize(image, tuple([self.resized_image_shape[1],
                                                                 self.resized_image_shape[0]]))) / 255.0)

            return (processed_image)

        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
            return (None)

    def get_label(self, prediction, labels):
        '''
        Given a prediction score list and labels, return the category that the image belongs to/Anomaly
        :param prediction: List of prediction scores (last layer output of model)
        :param labels:  Labels for the categories in order
        :return: Label of the image / Anomaly
        '''
        try:
            # Flatten the prediction list and find the index of the maximum element
            # Check if it is greater than a threshold, and give output accordingly
            prediction = np.array(prediction).flatten()
            max_prediction_ind = np.argmax(prediction)
            # print("max_prediction_ind is",max_prediction_ind)
            max_prediction = prediction[max_prediction_ind]
            if max_prediction < self.min_confidence:
                return ("No category (Anomaly)")
            else:
                return (labels[max_prediction_ind])

        except:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
            return (None)

    def predict(self):
        '''
        This method uses the model provided and the labels (in a specific format) to transform the input to an output
        Optionally, the prediction scores could also be outputted.
        Based on the kind of model used, different preprocessing to the image has to be done

        :param input: Input batch of (list of) image/data. The image will be scaled down and transformed according to the extra info
                    given along with the model
        :param model: Trained model used for prediction
        :param labels: Label to attach to the prediction (for human-readability)
        :return: give appropriate label to the image/data, with its prediction score
        '''
        try:
            response = {"image_class": [], "class_confidence": [], "error_msg": "All OK"}
            nn_id = self.temple_id
            input = self.input
            try:
                self.load_model(nn_id)
            except:
                response["error_msg"] = "Model does not exist/ Error in loading model"
                error_traceback = traceback.format_exc()
                self.error_logger(error_traceback)
                return (response)

            if self.current_model == None:
                response["error_msg"] = "Model does not exist/ Error in loading model"
                return (response)

            model = self.current_model
            labels = self.class_labels
            no_of_images = len(input)

            # Preprocessing all images
            input = np.array([self.preprocess_image(image) for image in input])

            # Apply model to each input and get the classes using the get_label() function
            predictions = model.predict(input)
            for prediction in predictions:
                response["image_class"].append(self.get_label(prediction, labels))
                response["class_confidence"].append(np.amax(prediction))

            print("response sent is", response)
            return (response)

        except Exception as error:
            error_traceback = traceback.format_exc()
            self.error_logger(error_traceback)
            return (None)

        pass

    def parse_query_file(self, query_file_path):
        # Opening query file and reading contents
        with open(query_file_path, 'r') as query_file:
            query_json = json.load(query_file)

        self.path_to_models = query_json["path to models"]
        query_images_folder = query_json["query images"]
        self.temple_id = query_json["temple id"]
        self.log_file_path = query_json["log file path"]

        imagePaths = list(paths.list_images(query_images_folder))

        # Creating the data numpy array using the paths
        data = []
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            data.append(image)

        self.input = data

    def set_paths(self, path_to_models, log_path):

        self.path_to_models = path_to_models
        self.error_log_file_path=os.path.join(log_path,"error_log.txt")
        self.status_log_file_path=os.path.join(log_path,"log.txt")

    def parse_query_json(self, query):
        self.temple_id = query["temple id"]
        imgdata = base64.b64decode(query["image"])
        filetype_str = query["image type"].strip('.')
        filename = 'query_image' + '.' + filetype_str
        with open(filename, 'wb') as f:
            f.write(imgdata)

        image = cv2.imread(filename)
        data = []
        data.append(image)

        self.input = data

    def check_database_for_queries(self):
        '''
        This method will continuously check a database for images that need to be processed.
        Once it finds images that need to be processed, it takes extra information, like the temple id
        All auxiliary information helps determine which neural network to use,etc.
        Generate a queries list with all the relevant information(preferably sorted with the NN id to be used)

        :return: A list of queries with the relevant information
        '''
        pass

    def traverse_queries(self, queries):
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

    # def preprocess_image(self,image):
    #     '''
    #     This method preprocesses the image before it is sent to the predict function
    #     eg. Turn to grayscale image, scale down th image, add dimensions to the image etc.
    #     :param image: image to be preprocessed
    #     :return: processed_image: preprocessed version of image
    #     '''
    #     pass

    def update_database_queries(self, queries_to_update):
        '''
        This method will edit the query records (corresponding to queries that have been processed)
        to state that they have been processed. Then the next run of check_database_for_queries will
        not consider those queries.

        :param queries_to_update: List of queries to mark as "processed" in the database
        :return:
        '''
        pass

    def ensure_model_present(self, nn_id):
        '''
        Given the neural network id, this method ensures that it is present in the memory. This can then be used
        for predictions. If the model is not present, then it will get the model from the database, or return an error
        (if the model is not trained/if it cannot be loaded for some reason)

        :param nn_id: Neural Network id of the required neural_network
        :return: True/False along with error code if necessary
        '''
        pass

    def error_logger(self, message):
        with open(self.error_log_file_path, 'a') as log_file:
            log_file.write("[Error Log at "+str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+" ]")
            log_file.write("\n")
            log_file.write(message)
            log_file.write("\n")

    def status_logger(self,message):
        with open(self.status_log_file_path, 'a') as log_file:
            log_file.write("[Status Log at "+str(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+" ]")
            log_file.write("\n")
            log_file.write(message)
            log_file.write("\n")
