# Gender Detection Project

## Overview
This project is focused on building a machine learning model to detect gender from images. Utilizing TensorFlow and OpenCV, the model is trained to classify images as either 'Male' or 'Female'.

## Dataset
The dataset used in this project is the CelebA dataset, which is a large-scale face attributes dataset with more than 200,000 celebrity images, each with 40 attribute annotations. The dataset can be found on Kaggle: [CelebA Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Requirements
- TensorFlow
- NumPy
- OpenCV
- Python 3.x

## Installation
To set up the project, follow these steps:
1. Install the required libraries: TensorFlow, NumPy, and OpenCV.
2. Download the CelebA dataset from Kaggle.
3. Clone this repository to your local machine.

## Usage
To use the model for gender detection, follow these steps:
1. Read the target image using OpenCV.
2. Resize the image to 150x150 pixels.
3. Convert the image into a NumPy array and reshape it as required by the model.
4. Load the pre-trained model 'model.h5'.
5. Use the model to predict the gender.
6. The output will be displayed as either 'Male' or 'Female'.

## Code Snippet
```python
import tensorflow as tf
import numpy as np
import cv2

# Reading and resizing the image
image = cv2.imread('path_to_image')
image = cv2.resize(image, (150, 150))
image = np.array(image)
image = image.reshape(1, 150, 150, 3)

# Loading the saved model
model = tf.keras.models.load_model('model.h5')

# Predicting the gender
dict = {1: 'Male', 0: 'Female'}
out_arr = model.predict(image)[0]
print(f'Predicted class is: {dict[np.argmax(out_arr)]}')

## Results and Discussion
The model performs gender classification with an accuracy of [insert accuracy]. While effective, it's important to note that gender is a complex and multi-dimensional concept that may not be fully captured by binary classifications.

## Future Work
Future improvements could include expanding the dataset, enhancing the model to recognize non-binary genders, and increasing the overall accuracy and robustness of the model.

## License
This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
