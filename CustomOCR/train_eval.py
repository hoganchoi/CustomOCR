## This module stores the training and evaluating process of the model.

## Import necessary packages.
import tensorflow as tf
import os
import sys
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import CustomOCR.ocr_model as om
import CustomOCR.utils.file_operations as fo
import CustomOCR.utils.image_operations as io

## Loads in training, validation, and testing datasets and preprocesses them for model training and evaluation.
def preprocess_training_images(train_path, val_path, test_path, case_sensitive = True):
    '''
    Loads in train, val, and test datasets while also preprocessing them to be compatible with model training
    and evaluation.

    Args:
        train_path (string): A string representing the directory for train file.
        val_path (string): A string representing the directory for validation file.
        test_path (string): A string representing the directory for test file.
        case_sensitive (boolean): Whether the english characters are case sensitive or not.

    Returns:
        train_data (np.array): An array containing all training images.
        train_labels (np.array): An array containing all training labels.
        val_data (np.array): An array containing all validation images.
        val_labels (np.array): An array containing all validation labels.
        test_data (np.array): An array containing all testing images.
        test_labels (np.array): An array containing all testing labels.
    '''
    ## Load in training, validating, and testing datasets from given paths.
    train_dict = fo.load_mat_data(train_path)
    val_dict = fo.load_mat_data(val_path)
    test_dict = fo.load_mat_data(test_path)

    ## Extract the images and labels from generated dictionaries.
    train_data, train_labels = io.stack_imgs_labels(train_dict)
    val_data, val_labels = io.stack_imgs_labels(val_dict)
    test_data, test_labels = io.stack_imgs_labels(test_dict)

    ## Expand the dimensions for each dataset to be compatible with model.
    train_data = np.expand_dims(train_data, axis = -1)
    val_data = np.expand_dims(val_data, axis = -1)
    test_data = np.expand_dims(test_data, axis = -1)

    ## Convert all labels to be compatible for categorical cross entropy for case sensitive labels.
    if case_sensitive:
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes = 62)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes = 62)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes = 62)

    ## Convert all labels to be compatible for categorical cross entropy for case insenstiive labels.
    if not case_sensitive:
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes = 36)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes = 36)
        test_labels = tf.keras.utils.to_categorical(test_labels, num_classes = 36)

    ## Return all images and labels.
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

## Given the path to training, validating, and testing datasets, trains model.
def train_model(train_path, val_path, test_path, out_path, batch_size = 32, case_sensitive = True, pre_weights = None):
    '''
    Trains the OCR model using the given mat files.

    Args:
        train_path (string): A string representing the directory for train file.
        val_path (string): A string representing the directory for validation file.
        test_path (string): A string representing the directory for test file.
        out_path (string): A string representing the directory for storing the trained weights.
        batch_size (int): The batch size to use when training the model.
        case_sensitive (boolean): Whether the english characters are case sensitive or not.
        pre_weights (string): A string representing the weights to a pre-trained model.

    Returns:
        train_history (History): History containing the historical loss and accuracy of model.
    '''
    ## Create all necessary datasets and labels.
    train_ds, train_labels, val_ds, val_labels, test_ds, test_labels = preprocess_training_images(train_path,\
                                                                                                  val_path,\
                                                                                                  test_path,\
                                                                                                  case_sensitive = case_sensitive)

    ## Define early stopping with patience of three epochs.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss', 
        patience = 3, 
        restore_best_weights = True
    )

    ## Define restoring best weights overall.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = out_path, 
        monitor = 'val_loss', 
        mode = 'min', 
        save_best_only = True, 
        save_weights_only = True
    )

    ## Initialize and compile OCR model.
    if case_sensitive:
        model = om.CustomOCRModel(pre_weights = pre_weights).generate_model()
    if not case_sensitive:
        model = om.CustomOCRModel(num_classes = 36, pre_weights = pre_weights).generate_model()

    ## Start the training process.
    print("Starting Model Training:\n")
    start_time = time.time()
    with tf.device('/GPU:0'):
        train_history = model.fit(
            train_ds, train_labels, 
            epochs = 15, 
            batch_size = batch_size, 
            shuffle = True, 
            validation_data = (val_ds, val_labels), 
            callbacks = [checkpoint]
        ) 
    end_time = time.time()
    print("Model Training Finished!\n\n")
    print(f"Time: {end_time - start_time}")

    ## Evaluate trained model on testing dataset.
    print("Starting Model Evaluation:\n")
    with tf.device('/GPU:0'):
        test_loss, test_acc = model.evaluate(test_ds, test_labels, batch_size = 32)
    ## Print out evaluation metrics.
    print("Model Evaluation Finished!\n")
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

    ## Return the training history.
    return train_history

## Saves trained model's metrics.
def save_model_metrics(train_history, out_dir):
    '''
    Saves mode's historical metrics.

    Args:
        train_history (History): History containing all metrics during model training.
        out_dir (string): A string representing the directory storing model and metrics.

    Returns:
        None
    '''

    ## The path for the metrics.
    out_metrics = os.path.join(out_dir, 'metrics')

    ## The path for training loss and accuracy.
    train_loss = os.path.join(out_metrics, 'training_loss.csv')
    train_acc = os.path.join(out_metrics, 'training_accuracy.csv')

    ## The path for validation loss and accuracy.
    val_loss = os.path.join(out_metrics, 'val_loss.csv')
    val_acc = os.path.join(out_metrics, 'val_accuracy.csv')

    ## Save training loss and accuracy to training path.
    fo.write_csv_file(train_history.history['loss'], train_loss)
    fo.write_csv_file(train_history.history['accuracy'], train_acc)

    ## Save validation loss and accuracy to validation path.
    fo.write_csv_file(train_history.history['val_loss'], val_loss)
    fo.write_csv_file(train_history.history['val_accuracy'], val_acc)

    print("All Files Saved Successfully!")