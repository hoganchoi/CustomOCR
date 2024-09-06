## This module stores all functions that preprocess an image so that they are compatible with the 
## OCR model.

## Import necessary packages.
import cv2
from PIL import Image
import numpy as np
import sys

## Make sure to set duplicates to true in order to apply CRAFT algorithm.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from craft_text_detector import Craft 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CustomOCR.utils.image_operations as io

import warnings
warnings.filterwarnings('ignore')

## Obtains the segmentation images for individual words in image.
def get_words(img_arr = None, img_path = None):
    '''
    Given an image with words, uses CRAFT algorithm to detect ROI encapsulating words.

    Args:
        img_arr (np.array): An array depicting an image with words.
        img_path (string): A string representing the image.

    Returns:
        word_imgs (List): A list containing all the images of the word.
        (List): A list containing the box coordinates encompassing the word.
    '''
    ## If the user inputted an image array, designate as the image variable.
    if img_arr is not None:
        ## Detect bounding box points using CRAFT algorithm.
        word_boxes = Craft().detect_text(image = img_arr)

        ## Designate the image array as img.
        img = img_arr

    ## If the user inputted an image path, open the image from image path using PIL.
    if img_path is not None:
        ## Detect bounding box points using CRAFT algorithm.
        word_boxes = Craft().detect_text(img_path)

        ## Load in the image array using PIL.
        img = Image.open(img_path)
        img = np.array(img)

    ## Initialize list for word images.
    word_imgs = []
    
    ## For each detected word, extract the image within bounding box.
    for word_box in word_boxes['boxes']:
        ## Obtain the bounding box's dimensions as well as coordinates.
        word_rect = cv2.minAreaRect(word_box)
        rect_coords = cv2.boxPoints(word_rect)
        rect_coords = np.int0(rect_coords)

        ## Find rotation angle.
        rect_center = (int(word_rect[0][0]), int(word_rect[0][1]))
        if word_rect[1][1] > word_rect[1][0]:
            rot_angle = word_rect[2] - 90
        else:
            rot_angle = word_rect[2]

        ## Un-rotate the image to make sure the word is horizontal.
        rot_mat = cv2.getRotationMatrix2D(center = rect_center, angle = rot_angle, scale = 1.0)
        rot_img = cv2.warpAffine(src = img, M = rot_mat, dsize = (img.shape[1], img.shape[0]))

        ## Extract the image within the bounding box.
        word_img = rot_img[min(rect_coords[:, 1]):max(rect_coords[:, 1]), 
                           min(rect_coords[:, 0]):max(rect_coords[:, 0])]
        
        ## Append word image to list.
        word_imgs.append(word_img)

    ## Return the list of word images and their respective coordinates.
    return word_imgs, word_boxes['boxes']

## Obtains the individual characters in a word image detected by CRAFT.
def get_char(word_imgs, text_bright = True):
    '''
    Segments individual characters from word images.

    Args:
        word_imgs (List): A list containing individual words detected in given image.
        text_bright (boolean): If the foreground is bright and the background is dark.

    Returns:
        word_list (List): A list containing characters for each respective word image.
    '''
    ## A list containing characters for each word image.
    word_list = []

    ## For each word image, segment characters.
    for word in word_imgs:
        ## Initialize character and indices list.
        char_list = []
        index_list = []

        ## Convert the image into greyscale and apply Gaussian Blur.
        img_grey = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(img_grey, (5, 5), 0)

        ## Apply Otsu's global thresholding to create binary mask image.
        _, binary_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)

        ## If the background is bright and the foreground is dark, invert the image to 
        ## correctly draw contours.
        if not text_bright:
            binary_img = cv2.bitwise_not(binary_img)

        ## Find contours of segmented characters.
        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ## Fill in the contours to make sure the insides are not detected.
        mask = np.zeros_like(img_grey)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        filled_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ## For each filled contour, extract box image containing character.
        for filled_contour in filled_contours:
            x, y, w, h = cv2.boundingRect(filled_contour)
            char_img = binary_img[y:y + h, x:x + w]
            char_list.append(resize_pad(char_img))

            ## If the characters are aligned horizontally, append x indices.
            if word.shape[1] >= word.shape[0]:
                index_list.append(x)
            ## If the characters are aligned vertically, append y indices.
            else:
                index_list.append(y)

        ## Get sorted images based on the placement of characters.
        sorted_char = io.sort_images(char_list, index_list)

        ## Append sorted characters into the list of words.
        word_list.append(sorted_char)

    ## Return the sorted character images.
    return word_list

## Resizes and pads images to be compatible with model.
def resize_pad(img):
    '''
    Resizes the image with respect to the longest side and then pads with background pixels.

    Args:
        img (np.array): An image array containing the image.

    Returns:
        mod_img (np.array): The modified image that is compatible with model.
    '''
    ## Get the width and height of the image.
    height, width = img.shape

    ## If the height is greater than the width, resize height to 28.
    if height > width:
        ## Get the resize factor based on height.
        resize_factor = 28 / height

        ## Resize the character image based on height.
        img_resize = cv2.resize(img, (0, 0), fx = resize_factor, fy = resize_factor)

        ## Set the padding needed for y-axis.
        pad_y = (2, 2)

        ## Determine the total padding needed for the x-axis.
        width_pad = 32 - img_resize.shape[1]

        ## Based on the value of the width pad, determine the amount of padding needed for the left and 
        ## right side of the image.
        if (width_pad) % 2 == 0:
            pad_x = (int(width_pad / 2), int(width_pad / 2))
        if (width_pad) % 2 == 1:
            pad_x = (int(width_pad // 2), int(width_pad // 2) + 1)

    ## If the width is greater than the height, resize width to 28.
    if width >= height:
        ## Get the resize factor based on width.
        resize_factor = 28 / width

        ## Resize the character image based on width.
        img_resize = cv2.resize(img, (0, 0), fx = resize_factor, fy = resize_factor)

        ## Set the padding needed for x-axis.
        pad_x = (2, 2)

        ## Determine the total padding needed for the y-axis.
        height_pad = 32 - img_resize.shape[0]

        ## Based on the value of the height pad, determine the amount of padding needed for the upper and
        ## lower side of the image.
        if (height_pad) % 2 == 0:
            pad_y = (int(height_pad / 2), int(height_pad / 2))
        if (height_pad) % 2 == 1:
            pad_y = (int(height_pad // 2), int(height_pad // 2) + 1)

    ## Pad the image accordingly.
    mod_img = np.pad(img_resize, (pad_y, pad_x), constant_values = 0)

    ## Return resized and padded image.
    return mod_img