import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# ===================== Step 1: Limit the Number of Images to 200 Per Class =====================

# Define path to your dataset directory
data_dir = 'hair_types'  # Original dataset directory

# Define the maximum number of images to retain in each class folder
MAX_IMAGES = 200

# Function to limit the dataset to 200 images per class
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

# Limit the dataset to 200 images for each class
limit_images(data_dir, MAX_IMAGES)

print(f"Dataset is now limited to {MAX_IMAGES} images for each class.")

# ===================== Step 2: Custom Preprocessing Function =====================

# Custom preprocessing with a less aggressive zoom effect
def custom_preprocessing(image):
    if np.random.random() < 0.2:  # Reduce the chance of zoom effects
        # Slightly scale up or down the image
        scale_factor = np.random.uniform(0.9, 1.1)  # Less zoom-in or zoom-out effect
        new_width = tf.cast(tf.cast(tf.shape(image)[1], tf.float32) * scale_factor, tf.int32)
        new_height = tf.cast(tf.cast(tf.shape(image)[0], tf.float32) * scale_factor, tf.int32)

        # Resize while keeping a reasonable aspect ratio
        image = tf.image.resize(image, (new_height, new_width))

        # Pad the image to keep dimensions at 150x150
        image = tf.image.resize_with_crop_or_pad(image, 150, 150)

    # Apply random flipping
    image = tf.image.random_flip_left_right(image)

    return image


# ===================== Step 3: Data Generators =====================

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    preprocessing_function=custom_preprocessing
)

# Load training dataset
train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
    interpolation='nearest'
)

# Load validation dataset
validation_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ===================== Step 4: Define the Model Architecture =====================

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
    Dense(5, activation='softmax')  # 5 classes for hair types
])

batch_size = 25

# Define steps per epoch
steps_per_epoch = len(train_generator) // batch_size

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("hairtype.keras", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

max_steps = 200

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# ===================== Step 5: Evaluate the Model =====================

# Evaluate the model on the validation dataset
val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // 32)
print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

# Generate classification report and confusion matrix
val_predictions = model.predict(validation_generator)
val_predictions = np.argmax(val_predictions, axis=1)
val_true_labels = validation_generator.classes

print("\nValidation Classification Report:")
print(classification_report(val_true_labels, val_predictions))

# Confusion Matrix for visualization
conf_matrix = confusion_matrix(val_true_labels, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Straight', 'Wavy', 'Curly', 'Dreadlocks', 'Kinky'],
           yticklabels=['Straight', 'Wavy', 'Curly', 'Dreadlocks', 'Kinky'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
model.save("hairtype.keras")

# ===================== Step 6: Prediction Function =====================

# Function to predict the class of a given image
def predict_image(image_path):
    model = tf.keras.models.load_model('hairtype.keras')
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    classes = ['Straight', 'Wavy', 'Curly', 'Dreadlocks', 'Kinky']
    return classes[predicted_class]

# Predict a sample image
image_path = 'hair_types/curly/sample.png'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")
