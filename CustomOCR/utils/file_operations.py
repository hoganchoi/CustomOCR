## Houses file operation functions that are used in other modules.

## Import necessary packages.
import mat73
import scipy.io
import hdf5storage
import os
import pandas as pd

## Saves a csv file given an output path directory.
def write_csv_file(file, file_path):
    '''
    Saves a csv file given a list.
    
    Args:
        file (List): A list containing elements.
        file_path (string): A string representing the output path directory.
    Returns:
        None
    '''
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    df_file = pd.DataFrame(file)
    df_file.to_csv(file_path, index = False, header = False)

## Load in mat file containing all the images and labels.
def load_mat_data(mat_path):
    '''
    Given the path to the mat file, loads in dictionary containing images and labels.

    Args:
        mat_path (string): A string representing the path to mat file.

    Returns:
        img_dict (dictionary): A dictionary containing all the images and labels.
    '''
    ## Load mat file using loadmat.
    ## First try loading mat file using scipy.io.
    try:
        mat_data = scipy.io.loadmat(mat_path)
    ## If mat file cannot be loaded in using scipy.io, use mat73.
    except TypeError:
        mat_data = mat73.loadmat(mat_path)

    ## If dictionary is from original mat data (contains gt key), extract only the images and labels.
    if list(mat_data.keys())[0] == 'gt':
        img_dict = mat_data['gt']
    ## If dictionary doesn't contain gt key, assign it as img_dict.
    else:
        img_dict = mat_data

    ## Return dictionary containing all images and labels.
    return img_dict

## Saves dictionary to given output path.
def save_mat_data(img_dict, output, use_scipy = True):
    '''
    Saves dictionary as a mat file to given directory.

    Args:
        img_dict (dictionary): A dictionary containing images and labels.
        output (stirng): A string representing the output file directory.
        scipy (bool): Determines whether to save mat file using scipy or hdf5storage.

    Returns:
        None
    '''
    ## Save dictionary using scipy.
    if use_scipy:
        scipy.io.savemat(output, img_dict)

    ## Save dictionary using hdf5storage (saves lists).
    if not use_scipy:
        hdf5storage.write(img_dict, '.', output, matlab_compatible = True)
    print("Mat file saved successfully!")

## Saves the model's weights to given output directory.
def save_model(model, output_path):
    '''
    Saves the trained model to output path.
    Args:
        model (model): A model that has been trained. 
        output_path (string): A string representing the output directory.
    Returns:
        None
    '''
    ## Get the directory name from the output path.
    directory = os.path.dirname(output_path)

    ## Check to see if directory exists. If not, create directory.
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## Save model weights to output path.
    model.save(output_path)
    print("Trained Model saved successfully!")