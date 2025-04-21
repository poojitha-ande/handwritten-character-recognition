#Importing all the necessary Libraries
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

#Reading the dataset
data = pd.read_csv('english.csv')
data

#Creating the lists and adding images into that empty lists 
images = []
labels = []

for index, row in data.iterrows():
    img_path = row['image']
    label = row['label']

    # Load the image
    img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img)
    images.append(img_array)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

#Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

labels = to_categorical(labels)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(images, labels, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(images, labels)
print(f'Test accuracy: {test_acc}')

#Plotting the graphs for model accuracy and model loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import random

#Predicting and to display the random images 
def predict_and_display_random_images(model, images, labels, label_encoder, num_images=1):
    random_indices = random.sample(range(len(images)), num_images)
    random_images = images[random_indices]
    actual_labels = labels[random_indices]

    # Predict the labels
    predictions = model.predict(random_images)
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_labels)
    actual_labels = label_encoder.inverse_transform(np.argmax(actual_labels, axis=1))

    num_rows = (num_images + 4) // 5
    plt.figure(figsize=(20, 4 * num_rows))
    
    #plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(num_rows, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(random_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(f"Actual: {actual_labels[i]}\nPredicted: {predicted_labels[i]}", fontsize=15)
    plt.show()

num_images = 1
predict_and_display_random_images(model, images, labels, label_encoder, num_images)




