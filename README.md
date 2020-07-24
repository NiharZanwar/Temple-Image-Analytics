# Temple-Image-Analytics
Train a Convolutional Network to Categorise Images and find anomalies

## CNN_temple_open_closed.py
This code takes training data as input and trains a CNN to learn to classify images as given in the training data.

This code also takes in a new dataset to (manually) test the categorisation on. It involves making folders labelled with class names and transferring (resized versions) images to the categories predicted by the model. It also outputs the images that failed to classify as any class (threshold value=0.5). These images are termed "anomalies", while the images that successfully classify as some class are called "normal" images.

Currently this model can satisfactorily classify between door_closed and door_opened images. Further experiments also assure that given enough pattern differences in the categories, the CNN (on training adequately) will classify well on more no. of categories as well.

## module: TempleImagesNN.py
This module provides a set of functions that can be called from other modules to train Convolutional Neural Networks.
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
_forceful_|Boolean value. If **true**, pretrained model will be overwritten with the new one. If **false** and a pretrained model exists, program will throw an error.
 
### check_for_trained_model()
Checks for a pretrained model for the same temple_id. If pretrained model exists, and *forceful* is **false**, then program will throw an error.

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
*image*|Base64 representation of the image to be categorised|Base64 string representing the image
*image_name*|Name of the image (Will be required to write the status log of the prediction)|String
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
*image_class*|Label given to the image based on the highest probability|String
*class_confidence*|Highest probability (corresponding to the probability of *image_class*)|Float


