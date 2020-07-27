# Temple-Image-Analytics
This project uses Convolutional Neural Networks(CNN's) from **Keras** as its core. 
With these programs you can train and manage several CNN's and use them to predict the category of new images. You can also
find out whether a given image is an anomaly (i.e. does not belong to any known categories).

**IMPORTANT**- Note that although this project has been created with the goal of categorising temple images, it can be used in any
general purpose image-categorisation scenario as CNN's (the base of this project) are general purpose.

Use cases include:
- Cameras fitted at several similar establishments (temples, banks, shops, playgrounds etc.) and several categories of images in each of these
establishments.
- Categorising the images of a single establishment in multiple ways (eg. "Door open/closed" and "Lights on/off")
- Any scenario in which one would want to save different "classifiers" and use them later on.

## How to use
A step-by-step guide on how to use this project.

1) Open command prompt/ terminal and install all the packages in the requirements.txt file. You can
do this by typing in the command ```pip install -r requirements.txt```
2) Determine the paths of the following directories and make a config file containing those paths. Add the
json file to config directory
   - training data
   - testing data
   - config
   - models
   - logs
   - app(Not required)
3) Make sure all the above directories have been created (preferably inside one single parent directory). 
Then, in the flask app, under the main method (under ```if __name__=="__main__"```) set the *path_to_config* to the path of the config file you
have generated.
4) Now, run the flask app. Hopefully, the app runs without error. If errors occur, one reason may be because you haven't created the
directories at the place specified in the config file.
5) Now, we should create some training and testing data for the CNN model to train on. Use the [save_data](#request-1-save-imagesdata) request in create_request.py or make your own request.
   - Refer to [Note: Training and Testing data](#note-training-and-testing-data) to understand how to provide data for good results.
6) Upload image and category data for a particular temple id. Do this for both training and testing data. Make sure that the training data and testing data
are different from each other.
7) Now we have training and testing data present in the respective folders, under the *temple_id* subdirectory.
8) Next, we will create a request to make a model on the image data we provided. Use  the [make_model](#request-2-make-model) request in create_request.py or make your own.
9) On making and sending the make_model request, the TempleNNTrainer object will start training a model. If errors occur,most of them
will be passed back in the form of a Response(json format) and an error log will be added. You can use these to make sure the training process runs smoothly.
10) If the training completes successfully (http code:200), then a saved model will be written (or overwritten) to **models**/**temple_id** directory.
11) Next, we will check our model's predictions on a new image. For this, use the [predict](#request-3-predict) request in create_request.py (or make your own request)
If the request is successful, you will get back a response with the category label with the highest probability 
(if no category has a higher probability than *min_confidence*, then the program returns "No category (Anomaly)") 
12) You can train several other models on data that you provide. All the models will be saved and will be available when you want to
predict the category of a new image.
13) If you are done using the flask app, you can shut it down by terminating the flask_server.py program.


## Modules used in this project
1) **keras**- This module is used to make, train and test the Convolutional Neural Networks.
2) **Tensorflow**- This module is used as a backend for **keras**.
3) **cv2**- OpenCV interface for Python. Used for handling images.
4) **sklearn**- (Scikit-learn) Module to help in the preprocessing of data for training the CNN.
5) **PIL**- (Python Imaging Library) Module used for image operations.
6) **imutils**- Convenience module for getting paths to images.
7) **os**- Module used for operations on directories and files.
8) **base64**- Module used to convert data to base64.
9) **json**- Module used to read json data format.
10) **numpy**- To handle numpy array operations.
11) **flask**- Module to make flask app.
12) **requests**- Module to make requests.
13) **traceback**- Used for logging errors.
14) **datetime**- Used for logging errors.

and other modules...

## Note: Training and Testing Data
In order to get good results for the prediction of the model, the points below will help
1) Make sure you have enough of data of a particular category. In my tests, around 100-200 images per category were sufficient.
If the model does not obtain good testing accuracy, and you have a lot of training data, consider increasing the number of epochs (currently 100) or tinkering with other parameters.
2) Make sure the data in a particular category is diverse enough. The training data should be representative of the kind of images the model is likely
to receive for prediction. Do not feed in too many images that are exactly the same. 
3) I found that having a common category of images called "None of the categories"(with random images), was very helpful in avoiding labelling random images as one of the categories.
   - The reason probably was that the CNN found some very small patterns that sufficed in segregating the images (like top-left corner is black/white,
   some specific pixels have some specific colour etc.)
   - Having "None of the categories images" with random images (eg. balloons, watches, animals, cars), probably helped in breaking those small patterns
   and forced the CNN to come up with larger patterns(hopefully the features that we humans would use to categorise)
   - Having diverse images for a single category could also help remove this problem.
   - Thus, the "None of the categories" also functions as an "anomaly" flag
4) If you want some category to be common to all of the models, add the category, with some image data to the **common** folder in the training data.
Currently, only the training data contains the common folder functionality. The program handles the common folder on its own and includes it in the training of all models.
   
## make_config_file.py
This module helps you to make a config file. This will be required for the working of the flask app.

The config files contains paths to all the important folders. The flask app will check for the existence of these folders before being available for post requests.

Query files are not implemented, so keep *create_query_flag*=False

## module: flask_server.py
This module runs a flask app. The flask app can:
1) Accept requests for uploading training/testing images to the appropriate folders (using the directory-subdirectory structure for the data images),
2) Accept requests to train CNN models based on the training and testing data already provided, and to save these models for later use, and
3) Accept requests to predict the category of a new image using pre-trained models

Currently, the requests can be posted to the particular routes only in the json format. The *create_request.py* module contains
code to make such requests to the flask app.

The flask app also checks for the availability of some base folders.

### main()
Once the flask app is created, the program checks for the existence of some folders, in order to work smoothly later on.

The main folder this program checks for is the config folder. The config folder should contain a config file that contains the paths
of other base folders listed below:
- training data : Contains training data
- testing data  : Contains testing data
- models    :   Contains saved models
- logs  :   Contains the error and status logs
- app   :   Should contain flask_server, TempleImagesNN, and all other required programs

If any of these folders (or the config file) does not exist, the program will terminate.

### app_route : /api/save_data
This route accepts a json request containing the following data

Key|Description|Datatype
---|---|---
*temple_id*|Temple id of the image that is being added|String
*train_test*|Whether image has to be added to training or testing directory|String ("train"/"test")
*category*|Category of the image.(The CNN model will learn from these categories)|String
*image*|Image to be saved encoded in base64|Base64 string representing an image
*image_type*|Filetype of the image.(Image will be saved as this filetype)|String (file extension eg. "jpg", "PNG" etc)
*image_name*|Name to be given to the image file|String

The program will then save the image sent as a request to the server, at the proper folder location, as mentioned by the keys
*temple_id*, *train_test* and *category*

A response will be sent back in json format. The json contains only one key i.e. error_msg.
- If the save is successful, a status code of 200 is sent, and the error_msg will read "All OK".
- If save is unsuccessful, a status code of 500 (server-side error) is sent, along with the error message in the error_msg key.

### app_route : /api/make_model
This route accepts a json request containing the following data

Key|Description|Datatype
---|---|---
*temple_id*|Temple id of the model to be trained|String
*forceful*|Boolean value whether you want to overwrite a pre-trained model if it exists|Boolean (True/False)

The program will:
1) Check if a pretrained model exists.
   - If yes, and *forceful* is False, throw error.
   - If yes, and *forceful* is True, continue with training.
   - If no, continue with training.
2) Load training ang testing data from the appropriate folders to start the training process.
3) Train the model on the training data.
4) Test the model on testing data.
5) Save the model to the *model_paths* directory and *temple_id* subdirectory.

A response is sent back that contains the status code 
- 200: request succeeded
- 400: bad request

as well as following information in json format:

Key|Description|Datatype
---|---|---
*error_msg*|List of error messages that occured during execution. Will be [] if request is successful.|List of Strings
*got_training_data_flag*|Boolean value whether training data was loaded properly|Boolean (True/False)
*got_testing_data_flag*|Boolean value whether testing data was loaded properly|Boolean (True/False)
*made_model_architecture_flag*|Boolean value whether model architecture was made|Boolean (True/False)
*trained_model_flag*|Boolean value whether model was trained properly|Boolean (True/False)
*tested_model_flag*|Boolean value whether model was tested properly|Boolean (True/False)
*saved_model_flag*|Boolean value whether model was saved properly|Boolean (True/False)

If an error occurs, the user can look at the flags to find out in which part did the error occur.
The flags, along with the error messages and error log(containing complete error traceback) will help the user debug quickly.

### app_route : /api/predict
This route accepts a json request containing the following data

Key|Description|Datatype
---|---|---
*temple_id*|Temple id of the image whose category has to be predicted|String
*image*|(list) Base64 representation of the image to be categorised|Base64 string representing the image
*image_name*|(list) Name of the image (Will be required to write the status log of the prediction)|String
*image_type*|Type of the image file. (Not used as prediction is done on the image without saving the image as a file first)|String (eg. "jpg","PNG" etc.)

The program will
1) Load the model from the *models* list.
   - If the model does not exist in the list, get the required model from the **models** directory, and then load it.
2) Preprocess the image, and feed it to the trained CNN model. Get the output probabilities for categories.
3) Using the output probabilities and the category labels, label the image with the category with highest probability.
(Give label "No category (Anomaly)" if no category has probability >=0.5)
4) Return a response containing the class label and the respective probability, along with error messages if errors occur.

A response is sent back that contains the status code 
- 200: request succeeded
- 400: bad request

as well as following information in json format:

Key|Description|Datatype
---|---|---
*error_msg*|List of error messages that occured during execution. Will be [] if request is successful.|List of Strings
*image_class*|(list) Label given to the image based on the highest probability|String
*class_confidence*|(list) Highest probability (corresponding to the probability of *image_class*)|Float

It is possible to give in a list of images rather than only single images. The (list) written in the tables mention that
 either single images, or multiple images could be given. Testing has only been done on single image queries, but the program has been
 made keeping batch image queries (all belonging to the same temple_id) in mind.
 
 Eventhough only the highest probability and its associated class are returned to the user, the status logs record all the probabilities.
 
 
## module: create_request.py
This module helps in creating requests and sending them to the Flask app in order to get a response back.
This program helps to test the working of the flask app, and also get an idea of how the app works.

It is important to note that in order to get a response back from the Flask app, it must be running in the first place.

Three kinds of requests can be made in this program. Towards the top, three flags corresponding to the three request types
can be set to True/False. If the flag is False, the request isn't sent, and if its True, the request is sent.

### Request 1: Save Images/Data
This request is sent if *request_save_data_flag* is True. The following variables can be configured for a save_data request:

Variable|Description
---|---
*save_data_folder_path*|Path of the folder containing categorised images
*save_data_temple_id*|Temple id of the images being saved
_save_data_image_type_|Filetype of the image being saved (eg. "jpg","PNG" etc.)
_save_data_train_test_|Whether the data is for training directory or testing directory.

A save images request baisically forms a copy of the *save_data_folder_path* in the **training** or **testing** directory.
All the categories are found out from the second-to-last element in the path (last element being the name of the image).

### Request 2: Make model
This request is sent if *request_make_model_flag* is True. The following variables can be configured for a make_model request:

Variable|Description
---|---
*make_model_temple_id*|Temple id of the images the model is to be trained on.
_make_model_forceful_|Whether a previous model has to be overwritten if it exists.

This request tells the flask app to train a new CNN model whose image data belongs to temple id=*make_model_temple_id*

If there is no error in the process, then the result will be a trained model that is saved in the **models**/**temple_id** directory.

### Request 3: Predict
This request is sent if *request_predict_flag* is True. The following variables can be configured for a predict request:

Variable|Description
---|---
*predict_temple_id*|Temple id of the image whose category is to be predicted.
_predict_image_path_|File path of the image.

This request tells the flask app to use a pre-trained CNN models to predict the category of a new image.


## module: TempleImagesNN.py
This module provides a set of functions that can be called from other modules (here they will be called in flask_server.py) to train Convolutional Neural Networks.
More info about the proper inputs and outputs will be updated as the module is built.


## _class : TempleNNTrainer_
**This class is a part of the TempleImagesNN module. It provides methods to train, test and save CNN models.**

**Explanation of the functions is given below**
#

### set_paths()
This function sets the paths to required folders and also sets the temple id for training. Parameters include

Parameters | Description
-----------|------------
_model_path_|Path containing all previously saved models. Used to check for pre-trained models and save new ones.
_training_data_path_|Path to the training data folder
_testing_data_path_|Path to the testing data folder (Required for training)
_log_path_|Path for log files. During training error/status logs will be written here
_temple_id_|Temple id of the temple. This will be appended to the training and testing paths to get the actual paths
_forceful_|Boolean value. If **True**, pretrained model will be overwritten with the new one. If **False** and a pretrained model exists, program will throw an error.
 
### check_for_trained_model()
Checks for a pretrained model for the same temple_id. If pretrained model exists, and *forceful* is **False**, then program will throw an error.

### preprocess_image()
Preprocesses input images to have the same shape and normalises pixel intensity values from 0-255 to 0-1.

First image in the training dataset is scaled using the *scale_factor* (eg. with *scale_factor*=*1/20*, (1920X1080)->(96,54)) , and others are resized to the shape of the resized first image.

### get_training_data()
Method to obtain training data from *training_data_path*. This method expects the training data in directory-subdirectory format.
- Directory = Temple id
- Subdirectories = Categories of images for the particular temple.

This method will also load common categories (path is set to *training_data_path*/*common*)

For this method to work without error, training data must be available with atleast 2 categories, and atleast 1 image in each category. (Although to train the model properly, about 200 images in each cateory must be provided).

### create_model_architecture()
Uses the **keras** *Sequential* class to create a CNN architecture.

Currently, model uses 2 convolutional layers with max_pooling and dropout=0.5 after each convolutional layer. The model then uses 3 fully connected layers, with the last layer having *number of nodes=number of categories*, with softmax activation.

If number of categories are large, we could try to increase complexity of the CNN architecture (eg. increase no. of convolutional layers/ fully connected layers)

### train_model()
Trains the model on the training data obtained previously. Training Parameters used:

Name|Value
-------|-------
Optimiser|Adam Optimiser
Learning Rate|1e-3 (=0.001)
Decay|(1e-3)/50 (=0.00002)
Loss| Categorical Crossentropy
Batch Size| 32
Epochs| 100
Early Stopping|*Not implemented for now*

### get_testing_data()
Method to obtain testing data from *testing_data_path*. This method expects the testing data in the same format as *get_training_data()*.

Common categories has not been implemented here, but may be done in the future.

It is important to note that testing data must be different from training data, else, even if we get a high accuracy on testing data, the model may not work well later on.

### test_model()
Tests the model that was cretaed and trained on the testing data. It checks if the model's predictions are consistent with the testing_data categories.

This method calculates the testing accuracy of the model. We can consider that the model can generalise well if it shows high accuracy on the testing dataset.

### save_model()
Saves the trained and tested model to the *model_path* directory with subdirectory named *temple_id*.

This method saves 3 files in the *temple_id* subdirectory:

Name|Type|Content
---|---|----
model_architecture.json|json file|Model architecture
model_weights.h5|h5 file|Trained Model weights
extra_info.json|json file|Contains (ordered) class labels and input image shape

### error_logger()
If an error is thrown in any of the above functions, this function is called. It writes the complete traceback of the error
in *log_path/error_log.txt* and also appends the error to the *list_of_errors* (implemented as last_error) in the class. The
*list_of_errors* will be used to form an appropriate response to a request (for more info read readme on flask_server.py).

### status_logger()
Any status updates are logged by this file. It writes the message in *log_path/log.txt*
Status updates like model trained, model tested etc. are entered here.

#

## _class : TempleNNPredictor_
**This class is a part of the TempleImagesNN module. It uses saved models to predict the category of new images.**

**Explanation of the functions is given below**
#
### get_model()
This method gets a saved model from *model_paths* and loads it into a list : *models*.
This method is used if a model is requested and it is not already present in the *models* list.

### load_model()
This method loads a requested NN model as the *current_model* along with other attributes such as *resized_image_shape* and *class_labels*
This *current_model* can then be used for predicting categories of an image.

Note- If a requested model is not present in the *models* list, the *get_model()* function is called.

### preprocess_image()
Similar functioning to *preprocess_image()* of the *TempleNNTrainer* class.

### get_label()
This method takes in the prediction probabilities (output of trained model) of 1 image, and returns the label corresponding to the highest
probability. 

This method uses a *min_confidence* parameter (currently hardcoded to 0.5) that specifies the minimum probability required for the
highest probability category for the method to output that category. In this example, if all categories have a probability score
less than 0.5 (less than *min_confidence*), the method will return "No category (Anomaly)"

The sum of the probabilities of all the categories is 1, so no two categories can have probability values greater than 0.5

### predict()
This is the most important method of the TempleNNPredictor class. It loads the required model (by calling the other methods),
gets the predicted category of the image, and sends back a *response*(json) to the caller.

This method has been tested on single image queries, but it could be tried on multiple image queries as well.

The *response* contains the image's predicted label, the confidence for that label, and error messages (if an error occurs during execution)

### set_paths()
This function is similar to the *set_paths()* function in *TempleNNTrainer*. Parameters include:

Parameters | Description
-----------|------------
_path_to_models_|Path containing all previously saved models. Used to load pre-trained models.
_image_names_|Names of all the images in a list. The names and predicted probabilities will be recorded in the status log.
_images_|All the images to run the prediction on, in a list
_log_path_|Path for log files. During training error/status logs will be written here
_temple_id_|Temple id of the temple. All *images* will have to belong to the same *temple_id*. This will be used to load the required model 

### error_logger()
This method has the same functionality of the *error_logger()* of the *TempleNNTrainer* class

### status_logger()
This method has the same functionality of the *status_logger()* of the *TempleNNTrainer* class

When a request is given to predict the category that an image belongs to, the status log records the probabilities of all categories.
In the response given back to the caller, only the highest class label and its probability are returned. Therefore, if we want more information/ troubleshoot a problem,
we can look at the status logs.


## CNN_temple_open_closed.py (NOT USED)
This code takes training data as input and trains a CNN to learn to classify images as given in the training data.

This code also takes in a new dataset to (manually) test the categorisation on. It involves making folders labelled with class names and transferring (resized versions) images to the categories predicted by the model. It also outputs the images that failed to classify as any class (threshold value=0.5). These images are termed "anomalies", while the images that successfully classify as some class are called "normal" images.

Currently this model can satisfactorily classify between door_closed and door_opened images. Further experiments also assure that given enough pattern differences in the categories, the CNN (on training adequately) will classify well on more no. of categories as well.
