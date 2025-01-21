import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from utils.load_data import load_train
from keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import load_img, img_to_array

ds = image_dataset_from_directory(
    "./dataset2",
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='grayscale',
    batch_size=64,
    image_size=(791, 552),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True
)

# Access the class names
#class_names = ds.class_names
#print("Class names:", class_names)

# Iterate through the dataset and decode labels
#for images, labels in ds:  # Take one batch
##    for label in labels:
#        class_name = class_names[label]
#        print(f"Label: {label.numpy()}, Class name: {class_name}")



num_classes = len(ds.class_names)


model = Sequential([
  layers.Rescaling(1./255, input_shape=(791, 552, 1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, padding='same', activation='softmax'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='softmax'),
  layers.Dense(num_classes)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


epochs=15
history = model.fit(
  ds,
  epochs=epochs
)

image_path = './dataset2/C#CC(=O)O/(2D-HMBC)_bmse000767_nmr_set01_1H_13C_HMBC_ser.png'

# Load the image (grayscale, size 300x210)
img = load_img(image_path, color_mode='grayscale', target_size=(791, 552))

# Convert the image to an array
img_array = img_to_array(img)

# Normalize pixel values to [0, 1]
img_array = img_array / 255.0

# Add a batch dimension
img_array = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_array)

# Convert logits to probabilities
probabilities = tf.nn.softmax(predictions[0])

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(probabilities)

# Get the class name
predicted_class_name = ds.class_names[predicted_class_index]

print(f"Predicted class index: {predicted_class_index}")
print(f"Predicted class name: {predicted_class_name}")
#print(predictions)