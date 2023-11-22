# Intern Data Science Task

## Table of Contents
* [Solution overview](#solution-overview)
* [Project Structure](#project-structure)
* [Installation and Running](#install-dependencies-and-run-the-program)
* [Model Training](#model-training)
* [Inference](#inference)
* [Detailed Description of Project Components](#detailed-description-of-the-project-and-its-modules)
  * [unet.py](#unetpy)
  * [unet_parts.py](#unet_partspy)
  * [airbus_dataset.py](#airbus_datasetpy)
  * [metrics.py](#metricspy)
  * [main.py](#mainpy)
  * [inference.py](#inferencepy)
  * [manual test images](#manual-test-images)
  * [trained model](#trained-model)
* [Data Balancing Strategy](#more-about-data-balancing-strategy)
* [Conclusion](#conclusion)


## Solution overview <a class="anchor" id="solution-overview"></a>

The model is based on the U-Net architecture using `tf.keras` and `DenseNet121` as the basis for the encoder. The model is trained with a customized loss function `Dice Loss` and evaluated by the `Dice Coefficient`.

## Project structure

The project includes the following main components:
- `unet.py`: Definition of the U-Net model.
- `unet_parts.py`: Model components such as decoder and convolution blocks.
- `airbus_dataset.py`: The `DataGenerator` class for data processing and image augmentation.
- `inference.py`: A script for manual testing of the model.
- `metrics.py`: Functions for calculating metrics and a loss function.
- `main.py`: The main script for training and saving the model.
- `manual_test_images`: A folder where the user places images for a manual test
- `trained_model`: Folder with the result of the trained model

## Install dependencies and run the program

To install the necessary dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Model training

To train the model, run the following command in the terminal:

```bash
python main.py
```

If you encounter an error like this one: `Unable to load image at path XXXXXXXXX.jpg`, please set the full path to the image directories and the error should disappear. Example: `C:/Project name/.../image_folder`

## Inference

To manually test the model, use this command in the terminal:

```bash
python inference.py
```

## Detailed description of the project and its modules

### `unet.py`

#### Implementation Details

- **Input Layer**: The model takes an input shape parameter that defines the dimensions of the input images.

- **Freezing Encoder Layers**: To retain the learned features from ImageNet and speed up training, we freeze the encoder layers during the training process.
  
- **Encoder**: Utilizes the pre-trained `DenseNet121` model as the encoder. In this architecture, specific layers have been selected to serve as feature extractors for the U-Net, particularly focusing on the pooling and activation layers. These layers were carefully chosen based on their ability to provide a rich feature set that captures a comprehensive representation of the input data, thereby offering optimal performance for our segmentation task. The pooling layers effectively reduce the spatial dimensions while preserving the most important features, and the activation layers introduce non-linearity to the feature maps, both of which are crucial for the model's predictive power.

- **Bridge**: Connects the encoder and decoder parts of the network.

- **Decoder**: Comprises a series of `decoder_block` functions imported from `unet_parts.py`. Each block upsamples its input and merges it with the corresponding encoder output (skip connection) to recover spatial information lost during downsampling.

- **Output Layer**: A convolutional layer with a `sigmoid` activation function that outputs the final segmentation map. The number of filters is set to the number of classes, which in the case of binary segmentation is 1.

- **Model Compilation**: The function `create_unet_model` consolidates all the layers into a `tf.keras.Model`, which is then returned.

#### Customizations

The U-Net model is customized to use DenseNet121 as the backbone for feature extraction. This choice is driven by the DenseNet architecture's ability to reuse features, reducing the number of parameters and computational burden.

#### Usage

This file is imported in `main.py` where the `create_unet_model` function is invoked to create the model instance for training and evaluation.

#### Why DenseNet121?

DenseNet121 is known for its feature reuse capabilities, which enhance the flow of information and gradients throughout the network, making it particularly suitable for segmentation tasks where both global and local features are important. Pre-training on ImageNet provides a robust set of features that can be fine-tuned for the specific task of ship detection.

### `unet_parts.py`

The `unet_parts.py` file defines essential building blocks for the U-Net model, specifically focusing on the decoder and convolutional operations that are pivotal in the model's architecture.

#### Implementation Details

- **Decoder Block**: The `decoder_block` function is designed to upsample the feature map and merge it with corresponding feature maps from the encoder, a process known as skip connection. This is crucial for the U-Net architecture as it allows the model to retain spatial information lost during the encoding (downsampling) process.

  - The function starts by upsampling the input feature map using `UpSampling2D`.
  - It then applies a `Conv2D` layer to the upsampled feature map to reduce the feature dimensions.
  - A concatenation operation follows, merging the upsampled feature map with the skip features from the corresponding encoder layer, allowing the model to use fine-grained details from earlier layers.
  - Finally, the merged feature map is passed through a convolutional block to refine the features further.

- **Convolutional Block**: The `conv_block` function defines a sequence of convolutional operations that are used within the decoder block.
  
  - It consists of two `Conv2D` layers, each followed by `BatchNormalization` and a `ReLU` activation.
  - This structure helps in stabilizing the learning process and introduces non-linearity, which is essential for learning complex patterns in the data.

#### Customizations

The blocks are modular and can be used repeatedly to construct the decoder part of the U-Net. By abstracting these operations into separate functions, the code is kept clean and maintainable, and the model's architecture becomes easier to understand and modify if needed.

#### Usage

These functions are imported and utilized in `unet.py` to construct the decoder pathway of the U-Net model. By separating these components, we can maintain a clear separation of concerns, which aligns with best practices in software development and deep learning model design.

### `airbus_dataset.py`

#### Data Processing and Image Augmentation

The `airbus_dataset.py` file implements the `DataGenerator` class, which is a key component for handling the loading, processing, and augmentation of image data for the model.

#### Implementation Details

- **Data Generator**: Inherits from `tf.keras.utils.Sequence` to ensure efficient loading and batch processing. It provides a robust mechanism to feed data into the model during training.

- **Initialization**: The generator takes a DataFrame containing image paths and mask information, the directory of images, batch size, and an augmentation flag to control whether data augmentation should be applied.

- **Resizing Images**: The original images have a resolution of 768x768 pixels. For training efficiency and to reduce memory load, images are resized to 512x512 pixels. This size offers a balance between model performance and computational efficiency.

- **Batch Processing**: Implements `__getitem__` to load and process images in batches, which is critical for training on large datasets. This method selects a batch of data, performs any necessary augmentation, and returns processed images and masks.

- **Image and Mask Loading**: The `_load_image_and_mask` method handles the loading and preprocessing of images and masks. Images are resized, and masks are decoded from their RLE (Run Length Encoding) format.

- **Data Augmentation**: To enhance the model's ability to generalize and reduce overfitting, data augmentation techniques such as rotation, flipping, brightness/contrast adjustments, and blurring/sharpening are applied randomly.

- **Shuffling Data**: The `on_epoch_end` method shuffles the data at the end of each epoch, ensuring that the model does not see data in the same order during each training cycle.

#### RLE Decoding

- The `rle_decode` function decodes the mask from its compressed RLE format to a binary mask. This step is crucial for preparing the masks for model training and evaluation.

#### Customizations

- The decision to resize the images to 512x512 pixels was made to balance between the detail level in images and the computational resources, as higher resolutions significantly increase the memory requirement.

#### Usage

This file is crucial in the model training process, as executed in `main.py`, where instances of `DataGenerator` are created for both training and validation datasets.

### `metrics.py`

#### Calculation of Metrics and Loss Function

The `metrics.py` file is dedicated to defining metrics and loss functions used to evaluate and optimize the U-Net model, specifically focusing on the Dice Coefficient and Dice Loss.

#### Implementation Details

- **Dice Coefficient**: 
  - Purpose: This function computes the Dice Coefficient, a common metric used in image segmentation tasks, particularly when handling imbalanced datasets. It measures the overlap between the predicted and true masks.
  - Implementation: The function flattens the binary mask arrays and calculates the intersection over the union. A `smooth` term is added to avoid division by zero and to provide stability in training.

- **Dice Loss**: 
  - Purpose: Defines a loss function based on the Dice Coefficient. In segmentation tasks, especially with imbalanced classes, Dice Loss can be more effective than traditional loss functions like cross-entropy.
  - Implementation: The Dice Loss is calculated as `1 - Dice Coefficient`. It is used to optimize the model during training, where a lower Dice Loss corresponds to a higher overlap between the predicted and actual masks.

#### Customizations

- The implementation of these metrics is tailored for segmentation tasks. The Dice Coefficient is particularly effective for the Airbus Ship Detection Challenge, as it directly evaluates the area of overlap, which is the primary interest in segmentation.
- The `smooth` parameter in the Dice Coefficient is a crucial addition to ensure numerical stability and to handle cases where masks may have no overlap at the beginning of training.

#### Usage

These functions are used in the `main.py` file for compiling the U-Net model. The Dice Coefficient serves as a metric to monitor during training and validation, while Dice Loss is used as the loss function to guide the optimization process.

### `main.py`

#### Main Script for Training and Saving the Model

The `main.py` file serves as the central script for orchestrating the training process of the U-Net model. It integrates various components of the project, from data loading to model training and saving.

#### Implementation Details

- **Data Preparation**: 
  - The script starts by defining paths to the training images and the CSV file containing segmentation data.
  - It loads the data using `pandas`, and processes it to identify images with and without ships, ensuring a balanced dataset.

- **Data Splitting**:
  - The dataset is split into training and validation sets, with stratification to maintain an equal ratio of images with and without ships in both sets.

- **Data Generators**:
  - Instances of the `DataGenerator` class are created for both training and validation datasets. These generators handle data loading, augmentation, and preprocessing.

- **Model Initialization**:
  - The U-Net model is instantiated using the `create_unet_model` function, with an input shape of 512x512 pixels.
  - The model is compiled using Adam optimizer and Dice Loss, with Dice Coefficient as the evaluation metric.

- **Training**:
  - The model is trained using the `fit` method on the training generator, with callbacks for saving the best model and checkpoints after each epoch.
  - The number of epochs, batch size, and other hyperparameters are defined here.

- **Model Saving**:
  - After training, the final model is saved to disk. This allows for later use in inference or further evaluation.

#### Customizations

- The choice of 512x512 as the input size balances the need for detailed image analysis and computational efficiency.
- Data balancing and augmentation techniques are crucial for training robust models, especially in tasks with imbalanced classes like ship detection.

#### Usage

Run this script to train the model from start to finish. It encapsulates the entire training process, making it straightforward to train the model with a single command:

```bash
python main.py
```

## More about Data Balancing Strategy

Due to the imbalanced nature of the dataset, with 22.1% of images containing ships and 77.9% without, a strategic approach to data sampling was required to train the model effectively.

![Dataset Imbalance](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/094dbded-a020-48be-8cbc-0cd34a35a772)

*Distribution of images in the dataset with and without ships*

### Explanation of the strategy

To eliminate the imbalance and prevent the model from shifting towards images without ships, the following data balancing method was applied:

- **All images with ships**: Every image with ships (22.1% of the dataset) was included in the training set to ensure that the model learns to detect ships with high accuracy.

- **Sample of images without ships**: To prevent the model from over-tuning to the more common class of images without ships, a number equal to half the number of images with ships was randomly selected from the "no ships" category (New Dataset: 67% images with ships, 34% images without ships).

This results in a training set where 33.15% of the original dataset is utilized. This proportion was chosen based on the rationale that detecting the absence of ships is generally easier and requires fewer examples for the model to learn this concept effectively.

### Rationale for the Chosen Proportion

The decision to train the model on 33.15% of the data aims to strike a balance between model performance and training efficiency. Training on images without ships is computationally less intensive, and including too many such images would lead to unnecessary training time without significant performance gains.

The graph above illustrates the distribution of images and underscores the reasoning behind our data sampling strategy, ensuring that the model is exposed to a balanced representation of both classes during training.

### `inference.py`

#### Script for Manual Model Testing

The `inference.py` script is used for performing inference on a set of images using the trained U-Net model. It demonstrates how the model can be applied to new data to generate predictions.

#### Implementation Details

- **Image Preprocessing**: 
  - The `preprocess_image` function reads an image from the file system, converts it to the RGB color space, resizes it to the expected input dimensions of the model (512x512), normalizes pixel values, and adds a batch dimension for model input compatibility.

- **Result Display**:
  - The `display_result` function visualizes the results in a two-panel plot, displaying the original image alongside the predicted mask for easy comparison.

- **Inference Process**:
  - The script loads the trained model from the specified path and iterates over all images in the given folder.
  - It preprocesses each image, performs prediction, applies a threshold to binarize the predicted mask, and then displays the result using `matplotlib`.

#### Customizations

- The decision to resize input images to 512x512 and the thresholding of prediction outputs at 0.5 are based on the model training configurations and the common practice in binary image segmentation tasks, respectively.

#### Usage

This script is designed to be run after the model has been trained and saved. To use the script, place the images you wish to test in the `manual_test_images` folder and execute the script as follows:

```bash
python inference.py
```

### `manual test images`

#### Folder for manual testing of the model

The `manual_test_images` folder is intended for uploading images to be used for manual testing of the model. This allows you to easily evaluate the model's performance on arbitrary data.

#### Usage

Users can place their images that they want to test with the model in this folder. After uploading the images to the `manual_test_images` folder, they can run the `inference.py` script to visualize the model's prediction results.

#### Testing process

When running the `inference.py` script, the model automatically processes all images in this folder and displays the original images alongside the generated masks. This allows users to evaluate how well the model is able to detect ships in the images.

#### Instructions for Users

Simply transfer the desired test images to the `manual_test_images` folder and run `inference.py`. Remember that the size of the images must be adequate for the model to process.

### `trained model`

#### Trained Model Directory

The `trained_model` directory contains the trained U-Net model after the completion of the training process. It houses a ready-to-use model that can be applied for inference or further evaluation.

#### Contents of the Directory

- The trained model is saved in this directory in the `.h5` format, which is the standard file format for Keras model serialization.
- The model file name usually includes information about the training epoch or performance indicators, which helps in identifying the specific iteration of training.

#### Usage

Users can load the trained model from this directory to perform inference on new data or to conduct an assessment of its effectiveness. This can be particularly useful for verifying the model's generalization capabilities on unseen data and for further fine-tuning.

## Conclusion

The work undertaken has culminated in the development of a model capable of detecting ships within images and generating corresponding masks for each vessel. The model performs admirably on images where ships are distinct and visible against the background. However, challenges arise with images that are heavily cluttered, such as busy port areas, or those with obstructive weather conditions like dense fog. Additionally, in scenarios where even the human eye might struggle to discern a ship, the model may not delineate the ship's form with high precision. In some cases, rather than producing exact rectangular masks, the model tends to create outlines that more closely resemble ovals, albeit still accurately indicating the location of the ships. Several examples of the model's testing outcomes will be provided to demonstrate its capabilities and limitations in various conditions. **All the data to demonstrate the model's operation was selected randomly**


![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/d109361d-59d3-494b-a457-0f71d9264e88)

*The image has clear visibility and the model has successfully generated a mask. However, the outline of the vessel shape is not perfect, but it accurately reflects the location of the vessel, but not its exact outline*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/8390398a-69c1-4f43-b980-a09ced3d8440)

*The image has clear visibility and the model has successfully generated a mask. However, the outline of the vessel shape is not perfect, but it accurately reflects the location of the vessel, but not its exact outline*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/30770a00-ebf5-48d2-851f-c4837e54249f)

*The model was able to see all the ships well, but there are some problem areas*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/476a47c5-e137-4314-bbb6-68c5c841bc20)

*The model saw a ship, but some of the waves were mistaken for a ship*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/4a16aaef-bd20-4816-a9ac-dc95d2b44639)

*Due to the high density of clouds or fog, the model could not see the ships*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/63e81d33-b2b6-45bc-9adb-4babeb1baf64)

*The model worked well and did not find any ships in the cloudy sky, so the mask is empty*

![image](https://github.com/Taras-Zelenyy/Intern-Data-Science-Task-/assets/83030264/0fe12b49-5cd6-4ffd-b992-531d1db4560e)

*In a dense port area, the model could not see moored ships*
