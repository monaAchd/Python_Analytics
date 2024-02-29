# -*- coding: utf-8 -*-
"""

@author: abhilash
"""

import os
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

folder_path = 'images/collected/fox'
true_labels = []
predicted_labels = []

# Load the model from internet / computer
# Approximately 96 MB
pretrained_model = InceptionV3(weights="imagenet")

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Get the image path
        img_path = os.path.join(folder_path, filename)

        # Load and preprocess the image
        img = load_img(img_path)
        img = img.resize((299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using the model
        prediction = pretrained_model.predict(img_array)
        actual_prediction = imagenet_utils.decode_predictions(prediction)

        # Extract the true label from the filename
        true_label = filename.split('.')[0].lower()  # Convert to lowercase for consistency
        true_labels.append(true_label)

        # Extract the predicted label
        predicted_label = actual_prediction[0][0][1]
        predicted_labels.append(predicted_label)

        # Print the prediction results
        print("Image:", filename)
        print("Predicted object is:", predicted_label)
        print("With accuracy:", actual_prediction[0][0][2] * 100)
        print("-----------------------")

        # Display image and the prediction text over it
        disp_img = cv2.imread(img_path)
        cv2.putText(disp_img, predicted_label, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
        cv2.imshow("Prediction", disp_img)
        cv2.waitKey(1000)  # Delay in milliseconds (1 second)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)
