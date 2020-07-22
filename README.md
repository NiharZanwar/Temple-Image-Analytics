# Temple-Image-Analytics
Train a Convolutional Network to Categorise Images and find anomalies

## CNN_temple_open_closed.py
This code takes training data as input and trains a CNN to learn to classify images as given in the training data.

This code also takes in a new dataset to (manually) test the categorisation on. It involves making folders labelled with class names and transferring (resized versions) images to the categories predicted by the model. It also outputs the images that failed to classify as any class (threshold value=0.5). These images are termed "anomalies", while the images that successfully classify as some class are called "normal" images.

Currently this model can satisfactorily classify between door_closed and door_opened images. Further experiments also assure that given enough pattern differences in the categories, the CNN (on training adequately) will classify well on more no. of categories as well.

## TempleImagesNN.py
This module provides a set of functions that can be called from other modules to train Convolutional Neural Networks.
More info about the proper inputs and outputs will be updated as the module is built.

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
 

