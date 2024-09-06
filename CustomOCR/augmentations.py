## Preprocess images for training model as well as augmenting images.

## Import necessary packages.
import imgaug.augmenters as iaa
import cv2
import random
import numpy as np
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CustomOCR.utils.file_operations as fo
import CustomOCR.utils.image_operations as io

## Resize all images to make it compatible with model.
def resize_imgs(images):
    '''
    Resizes all images to specific model.

    Args:
        images (np.array): An array (or list) containing all images.

    Returns:
        aug_imgs (np.array): An array of resized images.
    '''
    ## Resizes all images to be compatible with model.
    aug = iaa.Resize({"height": 32, "width": "keep-aspect-ratio"})

    ## Apply augmentations to array of images.
    aug_imgs = aug(images = images)

    ## Return augmented images.
    return aug_imgs

## Applies morphological alterations to images.
def erode_and_dilate(images):
    '''
    Randomly augments images in an array using erosion and dilation.

    Args:
        images (np.array): An array containing images for a certain instance (could be a list).

    Returns:
        aug_imgs (np.array): An array containing augmented images (could be a list).
    '''
    ## Create an array (or list) that will store augmented images.
    aug_imgs = images.copy()

    ## Define kernel for morphological augmentations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    ## For each image in original image array, randomly erode or dilate image.
    for i in range(len(images)):
        ## Get image for iteration.
        img = images[i]

        ## Apply erosion 25% of the time.
        if random.randint(1, 5) == 1:
            img = cv2.erode(img, kernel, iterations = random.randint(1, 2))

        ## Apply dilation 20% of the time.
        if random.randint(1, 6) == 1:
            img = cv2.dilate(img, kernel, iterations = random.randint(1, 1))
        
        ## Assign augmented image to augmented images array.
        aug_imgs[i] = img

    ## Return augmented images.
    return aug_imgs

## Augments images in a given array over multiple iterations.
def augment_images(images, iterations):
    '''
    Augments images over multiple iterations.

    Args:
        images (np.array): An array containing images for a certain class.
        iterations (int): The number of iterations augmentations are applied.

    Returns:
        np.array: An array containing unique augmented images.
    '''
    ## Initialize a list for augmented images.
    aug_list = list(images)

    ## Create augmentation sequence using imgaug.
    seq = iaa.Sometimes(
            1, 
            ## Apply a combination of the following augmentations.
            iaa.OneOf([
                ## Choose either Gaussian Noise or SaltAndPepper Noise.
                iaa.OneOf([
                    ## Choose one of Gaussian Noise or dropout.
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale = (0, 0.01 * 255)), 
                        iaa.Dropout(p = (0, 0.1))
                    ]),
                    ## Choose one of SaltAndPepper Noise or dropout.
                    iaa.OneOf([
                        iaa.SaltAndPepper(0.05), 
                        iaa.Dropout(p = (0, 0.1))
                    ])
                ]), 
                ## Choose a combination of transformations (shift, scale, translate, and rotate).
                iaa.SomeOf((1, None), [
                    iaa.Affine(rotate = (-10, 10)), 
                    iaa.Affine(scale = (0.5, 1.0)), 
                    iaa.Affine(translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                    iaa.Affine(shear = (-3, 3))
                ]),
                ## Apply Gaussian Blur.
                iaa.GaussianBlur(sigma = (0.0, 0.5))
            ])
        )
    
    ## Apply the given augmentations over specific number of iterations.
    for _ in range(iterations):
        aug_imgs = seq(images = images)
        aug_list.extend(aug_imgs)

    ## Convert the list in to an array.
    aug_array = np.array(aug_list)

    ## Return an array containing unique augmented images.
    return io.find_unique_imgs(aug_array)

## Augments the images given a dictionary.
def augment_dictionary(data_dict, aug_iter, augment = True):
    '''
    Applies augmentations to all images in dictionary.

    Args:
        data_dict (dictionary): A dictionary containing images and labels.
        aug_iter (int): The number of iterations for applying augmentations.
        augment (bool): Whether to apply augmentations or just resize.

    Returns:
        aug_dict (dictionary): Dictionary containing augmentated images.
    '''
    ## Create a copy from dictionary that'll store augmented images.
    aug_dict = data_dict.copy()

    ## Initialize images array.
    images = data_dict['images'][0]

    ## For each array containing images, apply augmentations.
    for i in range(len(images)):
        ## Specific array of images.
        class_imgs = images[i]

        ## Resize images based to model.
        resized_imgs = resize_imgs(class_imgs)

        ## If augmentation is true, apply augmentations.
        if augment:
            ## Erode and dilate.
            erode_dilate_imgs = erode_and_dilate(resized_imgs)
            ## Augment using transformations.
            aug_imgs = augment_images(erode_dilate_imgs, iterations = aug_iter)
            ## Save augmentations to new dictionary.
            aug_dict['images'][0][i] = aug_imgs

        ## If aumgentation is false, only save resized images.
        else:
            aug_dict['images'][0][i] = resized_imgs

        print(f"Finished Augmenting {i + 1} / {len(images)}")
    
    ## Return augmented dictionary.
    return aug_dict

## Creates and saves augmented dataset as a mat file.
def create_augment_dataset(input_dir, output_dir, aug_iter):
    '''
    Saves augmented dataset as a matfile.

    Args:
        input_dir (string): A string representing the directory for non-augmented mat file.
        output_dir (string): A string representing the directory for storing mat files.
        aug_iter (int): The number of iterations for applying augmentations.

    Returns:
        None
    '''
    ## Loads original mat file as a dictionary.
    mat_data = fo.load_mat_data(input_dir)

    ## Augments entire dictionary for specified iterations.
    aug_data = augment_dictionary(mat_data, aug_iter)

    ## Create the directory for output mat file.
    output_mat = os.path.join(output_dir, f'train_aug_{aug_iter}.mat')

    ## Save augmented dictionary to output directory.
    fo.save_mat_data(aug_data, output_mat)