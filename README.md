# Temple-Image-Analytics
Train a Convolutional Network to Categorise Images and find anomalies

## CNN_temple_open_closed.py
This code takes training data as input and trains a CNN to learn to classify images as given in the training data.

This code also takes in a new dataset to (manually) test the categorisation on. It involves making folders labelled with class names and transferring (resized versions) images to the categories predicted by the model. It also outputs the images that failed to classify as any class (threshold value=0.5). These images are termed "anomalies", while the images that successfully classify as some class are called "normal" images.

Currently this model can satisfactorily classify between door_closed and door_opened images. Further experiments also assure that given enough pattern differences in the categories, the CNN (on training adequately) will classify well on more no. of categories as well.

## TempleImagesNN.py
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


