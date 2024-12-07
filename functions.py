import tensorflow as tf
import numpy as np

#  Prediction function for testing purposes
def predict_skin(image_path):
    model = tf.keras.models.load_model('skincolor.keras')
    model.compile()
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    classes = ['dark', 'light', 'mid-dark', 'mid-light']
    return classes[predicted_class]

# Prediction function for testing purposes
def predict_hairtype(image_path):
    model = tf.keras.models.load_model('hairtype.keras')
    model.compile()
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    classes = ['curly', 'dreadlocks', 'Straight', 'Wavy', 'kinky']
    return classes[predicted_class]