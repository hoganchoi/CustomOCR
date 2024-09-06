## This module contains the script for augmenting the training dataset.

## Import necessary packages.
import os
import sys

## Import necessary modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CustomOCR.augmentations as aug

## The main function for this module.
def main():
    ## The input directory for training datasets.
    input_dir = '.\\datasets\\case_sensitive\\train_aug_0.mat'
    ## The output directory for augmented datasets.
    output_dir = '.\\datasets\\test'
    ## Augment the training dataset over 5 iterations. Please change the number of iterations to desired number.
    aug.create_augment_dataset(input_dir, output_dir, 5)

if __name__ == "__main__":
    main()