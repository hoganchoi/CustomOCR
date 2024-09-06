## This module stores the model for the optical character recognition.
## This includes initializing the modeland return predictions based on images.

## Import necessary packages.
from keras import Sequential, layers
from keras.optimizers import Adam
import numpy as np

## This class represents the CNN model for character recognition.
class CustomOCRModel():
    '''
    A class that contains the model for character recognition (case sensitive and insensitive).

    Attributes:
        input_shape (tuple): The input shape of the image.
        num_classes (int): The number of output classes (62 for sensitive 36 for insensitive).
        model (Sequential): The tensorflow model for character classification.
        loss (string): The string representing the loss function.
        metric (string): The string representing the metric.
        optimizer (optimizers): The optimizer used for training the model.
        pre_weights (string): The string representing the path for pre-trained weights.

    Methods:
        initialize_model (self): Initializes the model by laying out the model's architecture.
        load_model (self): If pre-trained weights were applied, load them in.
        generate_model (self): Compiles the model using the given loss function and optimizer.
        predict_char (self, char_img, labels): Given an image containing an english character and its respective labels, 
            predicts the character using the model.

    Usage:
        Used to generate and implement the OCR model. 
    '''
    ## Initializes the model.
    def __init__(self, num_classes = 62, pre_weights = None):
        '''
        Initializes the OCR model.

        Args:
            num_classes (int): The number of output classes for the model.
            pre_weights (string): The string representing the directory for the pre-trained weights.

        Returns:
            None
        '''
        self.input_shape = (32, 32, 1)
        self.num_classes = num_classes
        self.model = None
        self.loss = 'categorical_crossentropy'
        self.metric = 'accuracy'
        self.optimizer = Adam(learning_rate = 1e-4)
        self.pre_weights = pre_weights

    ## Initializes the model for OCR.
    def initialize_model(self):
        '''
        Creates the layout of the model that'll be used for OCR operations.

        Args:
            input_shape (tuple): The input shape of the images containing characters.
            num_classes (int): The number of characters that the model can recognize.

        Returns:
            None
        '''
        ## Create sequential linear stack for our CNN model.
        model = Sequential()

        ## Create the first 2D Convolutional layer.
        model.add(layers.Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', input_shape = self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha = 0.01))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))

        ## Create the second 2D Convolutional layer.
        model.add(layers.Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = 'valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha = 0.01))
        model.add(layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'valid'))

        ## Create the third 2D Convolutional layer.
        model.add(layers.Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'valid'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha = 0.01))

        ## Create 1x1 Convolutional layer for predicting abstract features within spatial orientation.
        model.add(layers.Conv2D(filters = 128, kernel_size = 1, strides = 1))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha = 0.01))

        ## Flatten the output from Convolutional layer.
        model.add(layers.Flatten())

        ## Create the fully-connected layer.
        model.add(layers.Dense(units = 1024))
        ## Add activation and dropout function.
        model.add(layers.LeakyReLU(alpha = 0.01))
        model.add(layers.Dropout(rate = 0.5))

        ## Create the final classifier layer.
        model.add(layers.Dense(units = self.num_classes, activation = 'softmax'))

        ## Save initialized model.
        self.model = model

    ## Generate model using pre-trained weights.
    def load_model(self):
        '''
        Loads in pre-trained weights.

        Args:
            pre_weights (string): A string representing the directory for pre-trained weights.

        Returns:
            None
        '''
        ## If model hasn't been initialized, initialize model.
        if self.model is None:
            self.initialize_model()
        
        ## Load in pre-trained weights to model.
        self.model.load_weights(self.pre_weights)

    ## Generates the model by adding loss, metrics, and optimizers.
    def generate_model(self):
        '''
        Compiles the initialized model.

        Args:
            None

        Returns:
            model (Sequential): The compiled model ready for training.
        '''
        ## Initialize the model.
        self.initialize_model()
        if self.pre_weights is not None:
            self.load_model()

        ## Compile the model using loss function and optimizer.
        self.model.compile(loss = self.loss, metrics = [self.metric], optimizer = self.optimizer)

        ## Return compiled model.
        return self.model

    ## Return prediction for a given image containing a character.
    def predict_char(self, char_img, labels):
        '''
        Given image with english character, return text prediction.

        Args:
            char_img (np.array): An image array containing an english character.
            labels (np.array): An array containing labels.

        Returns:
            char_pred (string): A string representing the predicted character from the model.
        '''
        ## Expand the dimensions of image to be compatible with model.
        char_img = np.expand_dims(char_img, axis = (0, -1))

        ## Get the predicted index in labels.
        label_indx = np.argmax(self.model.predict(char_img))

        ## Retrieve the string in labels given the index.
        char_pred = labels[label_indx]

        ## Return the predicted character in string format.
        return char_pred