import os
import zipfile
import tensorflow.keras as keras
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

folder_path = 'C:\Users\Dell\Downloads\images.zip\collected\wolf'
true_labels = []
predicted_labels = []

# Load the Xception model
pretrained_model = keras.applications.xception.Xception(weights="imagenet")

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Get the image path
        img_path = os.path.join(folder_path, filename)

        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.xception.preprocess_input(img_array)

        # Predict using the model
        prediction = pretrained_model.predict(img_array)
        predicted_label = keras.applications.xception.decode_predictions(prediction, top=1)[0][0][1]
        predicted_labels.append(predicted_label)

        # Print the prediction results
        print("Image:", filename)
        print("Predicted object is:", predicted_label)
        print("-----------------------")

        # Display image and the prediction text over it
        disp_img = cv2.imread(img_path)
        cv2.putText(disp_img, predicted_label, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0))
        cv2.imshow("Prediction", disp_img)
        cv2.waitKey(1000)  # Delay in milliseconds (1 second)

# Clean up
cv2.destroyAllWindows()

# Calculate the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)

