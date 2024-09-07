# USE ME
This is the `USEME.md` file. Here, I've documented how to properly set up Conda environment, run the available scripts, and use the CustomOCR class. 

## Setting Up Conda Environment
Please create and activate your Conda environment using the code below.

```markdown
conda create --name [name-of-your-virtual-environment] python=3.10
conda activate [name-of-your-virtual-environment]
```
(NOTE: Python 3.10 was used in order to install `Tensorflow=2.10.0`, which has GPU compatibility)

If you have a Nvidia GPU, please use the following commands to download `conda-forge`, `cudatoolkit`, and `cudnn`. This will allow you to tap into your GPU and access parallel computing when training your models.

```markdown
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

Now, please activate the Conda environment and download all the required packages from `requirements.txt` file using the following code.

```markdown
pip install -r requirements.txt
```

After installing all required packages, we can now run all the modules in this project.

## Running Scripts
There are only two scripts in this project: `augment_dataset` and `train_model`. By default, the `augment_dataset` script will load in the non-augmented training dataset for case-sensitive characters and augment them over 5 iterations. The `train_model` script will also load in the non-augmented training dataset for case-sensitive characters and save the trained models in `.\saved_models\case_sensitive\model_0` directory.

Before running the scripts, please make sure that your working directory is in the root folder of this project (something similar to below).

```markdown
C:\[path-to-your-projects]\OCR_Project\
```

The `augment_dataset` script will look something like below.

```python
def main():
    input_dir = '[directory-to-training-dataset]'
    output_dir = '[directory-to-augmented-training-dataset]'
    ## Augment given dataset over n iterations and save to output directory.
    aug.create_augment_dataset(input_dir, output_dir, n)
```

The `input_dir` should be the directory to your training dataset (case-sensitive or case-insensitive) and the `output_dir` should be the directory where you store all your augmented datasets. The `create_augment_dataset` function will augment the entire dataset $n$ times, and then remove any duplicate images. You can change the $n$ variable based on how large or diverse your current training dataset is.

The `train_model` script will be formatted similarly to code shown below.

```python
def main():
    train_path = '[directory-to-training-dataset]'
    val_path = '[directory-to-validating-dataset]'
    test_path = '[directory-to-testing-dataset]'

    ## If you want to do transfer learning, load in pre-trained weights.
    weights_path = '[directory-to-pre-trained-weights]' 

    model_output = '[directory-to-saved-models]'
    metric_output = '[directory-to-saved-metrics]'

    ## Trains the model using the specified parameters (please change accordingly).
    history = te.train_model(train_path, val_path, test_path, model_output,
                             case_sensitive = True, batch_size = 32, 
                             pre_weights = weights_path)

    ## Saves the historical metrics to metric_output.
    te.save_model_metrics(history, metric_output)
```

The `train_path`, `val_path`, and `test_path` are the directories to your training dataset, validating dataset, and testing dataset, respectively. If you want to perform transfer learning with pre-trained weights, please specify the `weights_path` parameter. The `model_output` and the `metric_output` are the directories storing your saved models and historical training metrics (loss, accuracy, etc).

The `train_model` function will train the custom OCR model using the determined parameters. By default, the model will be training on case-sensitive mode with a batch size of 32. However, you can change these parameters to best fit.

After changing the directories or using default directories, you can run the scripts by using the following commands:
 - ```markdown
    ## Run the augmentation script.
    python C:\[path-to-your-projects]\OCR_Project\scripts\augment_dataset.py
   ```
 - ```markdown
    ## Run the training script.
    python C:\[path-to-your-projects]\OCR_Project\scripts\train_model.py
   ```

## Using Notebooks
There are four notebooks in the `demos` folder. Three of the notebooks (`preprocess_demo`, `augmentations_demo`, and `model_training_demo`) are technical demonstrations of the individual steps taken to create the custom OCR system. The `custom_ocr_demo` notebook is a practical demonstration of how to use the custom OCR system. Below are the description for each notebook:
 - `preprocess_demo`: This notebook covers how to use the CRAFT algorithm to identify words in images and apply the threshold segmentation to extract individual characters. Also goes over image resizing and padding for model compatibility.
 - `augmentations_demo`: Describes all the different types of morphological augmentations applied to the training datset. Furthermore, discusses the properties of the operations from the `augmentations` module in `CustomOCR`.
 - `model_training_demo`: Provides a layout on training the model using case-sensitive, non-augmented training data. Also covers how to convert original data to be compatible with Tensorflow model training.
 - `custom_ocr_demo`: A demonstration of the CustomOCR system developed in this project. Applies the model to two different images and covers the performance of case-sensitive and case-insensitive characters.

(NOTE: The notebook documents the results compiled by my personal computer. Hence, the results documented in the notebook are historical and you may not get the same results. Additionally, because the training, validating, and testing images are not initially provided, please don't run the notebook unless you have the necessary datasets. You can download these images in the README file.)

## CustomOCR Documentation
The CustomOCR class in the `custom_ocr` module combines all the modules in this project to create a working OCR system. Below gives the documentation for initializing the class and its methods.

**Initializing CustomOCR:**
```python
CustomOCR(image, text_bright, case_sensitive, weights_path)
```
Initializes the CustomOCR class. 

*Parameters:*
 - `image`: An image array or image path consisting of text.
 - `text_bright`: If the text is brighter than the background.
 - `case_sensitive`: Whether or not to consider case sensitivity.
 - `weights_path`: The path to any pre-trained weights (by default, CustomOCR uses the model trained on the dataset augmented over 10 iterations for both case-sensitive and case-insensitive).

*Returns:*
None

**Extracting Text:**
```python
CustomOCR().extract_text()
```
Identifies and converts the words in an image to a machine-readable text format.

*Parameters:* None

*Returns:*
 - `pred_words`: A list containing all the predicted words in the image.
 - `box_coords`: A list of all the bounding box coordinates for each detected word.

Please look at the `custom_ocr_demo` notebook in the `demos` folder for more information on how to implement CustomOCR.