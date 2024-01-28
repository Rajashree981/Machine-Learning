import numpy as np
import tensorflow as tf
import cv2 
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
#loading image
image_name  = 'fn.jpeg'
img = cv2.imread(image_name)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
#image transformation
img_resized = cv2.resize(img_rgb, (224,224))
img_processed = preprocess_input(img_resized)
img_ready = np.expand_dims(img_processed, axis = 0)
#import model
model = EfficientNetB0(weights = 'imagenet')
#predict image
predictions = model.predict(img_ready)
dec_predictions = decode_predictions(predictions, top=5)
print("Predictions for the image")
plt.imshow(img_rgb)
plt.title(f'Pred:-{dec_predictions[0][0][1]} {dec_predictions[0][0][2]:.2f}')
dec_predictions
