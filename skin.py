import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from pathlib import Path

import os
import shutil

# Define path to your dataset directory
data_dir = 'data_skintone 2'

# Define the maximum number of images per class
MAX_IMAGES = 6000

# Function to limit images to MAX_IMAGES per class directory
def limit_images(directory, max_images):
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if len(images) > max_images:
                print(f"{class_name} has {len(images)} images, limiting to {max_images}")
                # Only keep the first `max_images` images
                for img in images[max_images:]:
                    os.remove(os.path.join(class_path, img))

# Limit the dataset to 6000 images for each class
limit_images(data_dir, MAX_IMAGES)


# Custom preprocessing with random sizing to account for different people's nail sizes and appearances
def custom_preprocessing(image):
    if np.random.random() < 0.3:
        # Randomly choose stretching or condensing
        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(1.1, 1.3)
        else:
            scale_factor = np.random.uniform(0.7, 0.9)

        new_width = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * scale_factor, tf.int32)
        new_height = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale_factor, tf.int32)
        image = tf.image.resize(image, (new_height, new_width))
        image = tf.image.resize_with_crop_or_pad(image, 150, 150)
        image = tf.image.random_flip_left_right(image)

    return image

# Load images from directories
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=custom_preprocessing
)

train_generator = train_datagen.flow_from_directory(
    directory='data_skintone 2',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Change to 'categorical' for multi-class classification
    subset='training',
    shuffle=True,
    seed=42,
    interpolation='nearest'
)

validation_generator = train_datagen.flow_from_directory(
    directory='data_skintone 2',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the model with a softmax output layer for 5 classes
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(512, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.4),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 5 classes with softmax activation
])

batch_size = 25

# Define steps per epoch
steps_per_epoch = len(train_generator) // batch_size

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("skincolor.keras", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

max_steps = 150

# Compile the model with categorys
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // 32)
print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

# Print classification report for multi-class evaluation
val_predictions = model.predict(validation_generator)
val_predictions = np.argmax(val_predictions, axis=1)
val_true_labels = validation_generator.classes

print("Validation Classification Report:")
print(classification_report(val_true_labels, val_predictions))

# Confusion Matrix for visualization
conf_matrix = confusion_matrix(val_true_labels, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=['dark', 'light', 'mid-dark', 'mid-light'],
           yticklabels=['dark', 'light', 'mid-dark', 'mid-light'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
model.save("skincolor.keras")

# Prediction function for testing purposes
def predict_skin(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    classes = ['dark', 'light', 'mid-dark', 'mid-light']
    return classes[predicted_class]

# Example usage
image_path = 'hair_types/curly/image.png'
prediction = predict_skin(image_path, model)
print(f"Predicted class: {prediction}")
