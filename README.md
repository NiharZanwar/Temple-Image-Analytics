# Temple-Image-Analytics
Train a Convolutional Network to Categorise Images and find anomalies

## CNN_temple_open_closed.py
This code takes training data as input and trains a CNN to learn to classify images as given in the training data.

This code also takes in a new dataset to (manually) test the categorisation on. It involves making folders labelled with class names and transferring (resized versions) images to the categories predicted by the model. It also outputs the images that failed to classify as any class (threshold value=0.5). These images are termed "anomalies", while the images that successfully classify as some class are called "normal" images.

Currently this model can satisfactorily classify between door_closed and door_opened images. Further experiments also assure that given enough pattern differences in the categories, the CNN (on training adequately) will classify well on more no. of categories as well.
