## This module stores the OCR processor. Here, users can import the CustomOCR class and 
## extract text from images. 

## Import necessary packages.
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CustomOCR.preprocess as pre
import CustomOCR.ocr_model as om

import warnings
warnings.filterwarnings('ignore')

## This class represents the entire OCR processor.
class CustomOCR():
    '''
    A class that is able to do the full Optical Character Recognition process.

    Attributes:
        image (np.array or string): An image array or the file directory for the image.
        text_bright (boolean): Whether the image has a dark background and bright text.
        weights_path (string): The file directory containing the pre-trained weights.
        model (CustomOCRModel): The CNN model for character recognition.
        labels (np.array): An array containing the labels.
        words (List): A list containing the words detected by CRAFT algorithm.
        box_coords (List): A list containing the coordinates for each detected word.
        chars (List): A list of segmented characters for each detected word.
        pred_words (List): The list of the predicted words from the model.

    Methods:
        extract_text (self): Given an image, locates the words and convert them into string based text.

    Usage:
        Locate and convert words in images to text.
    '''
    ## Initializes the OCR system.
    def __init__(self, image, text_bright = True, case_sensitive = True, weights_path = None):
        '''
        Given the image and necessary parameters, initializes the OCR system.

        Args:
            image (np.array or string): An image array or the file directory for the image.
            text_bright (boolean): Whether the image has a dark background and bright text.
            case_sensitive (boolean): Whether the predicted characters are case sensitive or not.
            weights_path (string): The path to the pre-trained weights.

        Returns:
            None
        '''
        ## Store image and whether the text is bright or not.
        self.image = image
        self.text_bright = text_bright

        ## If weights_path parameter is none, load in default weights.
        if weights_path is None:
            ## Load in case sensitive weights.
            if case_sensitive:
                self.weights_path = ".\\saved_models\\case_sensitive\\model_2\\model_2.h5"
            ## Load in case insensitive weights.
            if not case_sensitive:
                self.weights_path = ".\\saved_models\\case_insensitive\\model_2\\model_2.h5"
        ## If weights_path parameters is not note, load in inputted weights.
        else:
            self.weights_path = weights_path
            
        ## Load in model based on case sensitivity.
        if case_sensitive: 
            self.model = om.CustomOCRModel(pre_weights = self.weights_path)
        if not case_sensitive:
            self.model = om.CustomOCRModel(num_classes = 36, pre_weights = self.weights_path)
        self.model.load_model()

        ## Initialize the model's labels based on case sensitivity.
        if case_sensitive:
            self.labels = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'a', 'B',
                                    'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h',
                                    'I', 'i', 'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O',
                                    'o', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't', 'U', 'u',
                                    'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'], dtype='<U1')
        if not case_sensitive:
            self.labels = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                                    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], dtype='<U1')

        ## Initialize the extracted words and the predicted words.
        self.words = None
        self.box_coords = None
        self.chars = None
        self.pred_words = []

    ## Extracts and converts all words in given image to text based.
    def extract_text(self):
        '''
        Given an image with words, converts them into string based text.

        Args:
            None

        Returns:
            self.pred_words (List): The list of predicted words that are present in the image.
            self.box_coords (List): The list of coordinates for each predicted word.
        '''
        ## Extract bounding boxes for each word based on if the image is an array or a file directory.
        if isinstance(self.image, np.ndarray):
            self.words, self.box_coords = pre.get_words(img_arr = self.image)
        if isinstance(self.image, str):
            self.words, self.box_coords = pre.get_words(img_path = self.image)

        ## Segment the characters in each word location.
        self.chars = pre.get_char(self.words, self.text_bright)

        ## Initialize count.
        count = 1

        ## Predict individual characters and combine them to create a full predicted word.
        for word in self.chars:
            ## Initialize the predicted word.
            pred_char = ''
            
            ## Predict each character and append them to the predicted word.
            print(f'Predicting Word {count}:\n')
            for char_img in word:
                pred_char = pred_char + self.model.predict_char(char_img, self.labels)
            self.pred_words.append(pred_char)
            count = count + 1
        
        print("Finished Predictions!")
        
        ## Return the list of predicted words and their bounding box coordinates.
        return self.pred_words, self.box_coords