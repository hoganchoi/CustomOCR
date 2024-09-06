## This module contains the script for training the model.

## Import necessary packages.
import os
import sys

## Import necessary modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import CustomOCR.train_eval as te

## The main function for this module.
def main():
    ## The input directory for training datasets.
    train_path = '.\\datasets\\case_sensitive\\train_aug_0.mat'
    ## The input directory for validation datasets.
    val_path = '.\\datasets\\case_sensitive\\val.mat'
    ## The input directory for testing datasets.
    test_path = '.\\datasets\\case_sensitive\\test.mat'
    ## The input directory for pre-trained weights, if needed.
    weights_path = None

    ## The output directory for model's weights.
    model_output = '.\\saved_models\\case_sensitive\\model_0\\model_0.h5'
    ## The output directory for the training metrics.
    metric_output = '.\\saved_models\\case_sensitive\\model_0'

    ## Train the model for case sensitive datasets using the given paths.
    history = te.train_model(train_path, val_path, test_path, model_output, case_sensitive = True, batch_size = 32, pre_weights = weights_path)

    ## Save the historical metrics.
    te.save_model_metrics(history, metric_output)

if __name__ == "__main__":
    main()