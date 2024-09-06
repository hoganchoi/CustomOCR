## Houses image operations that are used in other modules.

## Import necessary packages.
import numpy as np
import copy
from sklearn.model_selection import train_test_split

## Given an array of images, removes all duplicates.
def find_unique_imgs(images):
    '''
    Removes all duplicates in an array of images.

    Args:
        images (np.array): An array containing images.

    Returns:
        unique_imgs (np.array): An array containing images without duplicates.
    '''
    ## Flatten the images array into vectors.
    flatten_imgs = images.reshape(images.shape[0], -1)

    ## Removes any duplicate vectors by viewing the first axis.
    unique_vectors = np.unique(flatten_imgs, axis = 0)

    ## Reconstruct flattened vectors into images.
    unique_imgs = unique_vectors.reshape(unique_vectors.shape[0], images.shape[1], images.shape[2])

    ## Return arrays with images without any duplicates.
    return unique_imgs

## Create training and testing datasets given data dictionary.
def create_train_test(data_dict, test_split = 0.2):
    '''
    Splits the dictionary into training and testing dictionaries.

    Args:
        data_dict (dictionary): A dictionary storing all images.
        test_split (float): The size for the testing dataset.

    Returns:
        train_dict (dictionary): A dictionary containing the training images.
        test_dict (dictionary): A dictionary containing the testing images.
    '''
    ## Deepcopy the data dictionary to training and testing dictionaries.
    train_dict = copy.deepcopy(data_dict)
    test_dict = copy.deepcopy(data_dict)

    ## Initialize the images.
    images = data_dict['images'][0]

    ## For each image class, apply the training testing split.
    for i in range(len(images)):
        ## Get the image array containing all images for specific class.
        img_arr = images[i].copy()
        ## Create training and testing images.
        train_arr, test_arr = train_test_split(img_arr, test_size = test_split, random_state = 42)

        ## Assign training and testing images to respective dictionaries.
        train_dict['images'][0][i] = train_arr
        test_dict['images'][0][i] = test_arr

    ## Return training and testing dictionaries.
    return train_dict, test_dict    

## Create training and validation datasets given data dictionary.
def create_train_val(data_dict, train_split = 0.2):
    '''
    Splits the dictionary into training and validation dictionaries.

    Args:
        data_dict (dictionary): A dictionary storing all images.
        train_split (float): The size for the validation dataset.

    Returns:
        train_dict (dictionary): A dictionary containing the training images.
        val_dict (dictionary): A dictionary containing the validation images.
    '''
    ## Deepcopy the data dictionary to training and validating dictionaries.
    train_dict = copy.deepcopy(data_dict)
    val_dict = copy.deepcopy(data_dict)

    ## Initialize the images.
    images = data_dict['images'][0]

    ## For each image class, apply the training validating split.
    for i in range(len(images)):
        ## Get the image array containing all images for specific class.
        img_arr = images[i].copy()
        ## Create training and validating images.
        train_arr, val_arr = train_test_split(img_arr, test_size = train_split, random_state = 42)

        ## Assign training and validation images to respective dictionaries.
        train_dict['images'][0][i] = train_arr
        val_dict['images'][0][i] = val_arr

    ## Return training and validation dictionaries.
    return train_dict, val_dict

## Flattens the images into one vector and create corresponding labels.
def stack_imgs_labels(data_dict):
    '''
    Creates a one dimensional vector for images and labels.

    Args:
        data_dict (dictionary): A dictionary containing images and labels.

    Returns:
        np.array, np.array: A flattened array containing images and an array of labels.
    '''
    ## Initialize empty list for labels.
    combined_labels = []
    
    ## For each array containing images in data_dict, create the same number of labels.
    for i in range(len(data_dict['images'][0])):
        ## Get image for iteration.
        images = data_dict['images'][0][i]

        ## Create array storing the same number of labels as images.
        labels_arr = np.full((images.shape[0], 1), i, dtype = np.float32)

        ## Append to list of labels.
        combined_labels.append(labels_arr)

    ## Return flattened array of images and labels.
    return np.vstack(data_dict['images'][0]), np.vstack(combined_labels)

## Sorts images based on their indices.
def sort_images(img_list, index_list):
    '''
    Given a list of images and their corresponding index, sorts the list.

    Args:
        img_list (List): A list containing images.
        index_list (List): A list containing corresponding indices.

    Returns:
        sorted_images (List): A list containing sorted images.
    '''
    ## Initialize list for sorted images.
    sorted_images = []

    ## Pair the image list and index list.
    img_index = list(zip(index_list, img_list))
    ## Sort the paired list based on index.
    sorted_list = sorted(img_index, key = lambda x: x[0])

    ## Save the sorted images into list.
    for index, image in sorted_list:
        sorted_images.append(image)

    ## Return list of sorted images.
    return sorted_images
