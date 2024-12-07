import tensorflow as tf
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Example usage for recoloring "shortwave.jpg"
image = 'eddie_teter.jpeg'

# Prediction function for skin tone
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

# Prediction function for hair type
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

# Recolor the image based on predictions
def recolor_image(input_image_path, output_image_name, skin_color, hair_color):
    """
    Recolor an image based on predicted skin tone and hair type.

    Args:
        input_image_path (str): Path to the input image.
        output_image_name (str): Name of the output image file.
        skin_color (tuple): RGB values for skin areas.
        hair_color (tuple): RGB values for hair areas.

    Returns:
        str: Path to the saved image in the downloads folder.
    """
    # Load the image
    image = cv2.imread(input_image_path)

    # Convert the image to HSV for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define masks for red (skin) and green (hair) colors
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply the masks
    result_image = image.copy()
    result_image[red_mask > 0] = skin_color  # Apply skin color
    result_image[green_mask > 0] = hair_color  # Apply hair color

    # Create a downloads folder if it doesn't exist
    downloads_folder = "downloads"
    os.makedirs(downloads_folder, exist_ok=True)

    # Save the image in the downloads folder
    output_image_path = os.path.join(downloads_folder, output_image_name)
    cv2.imwrite(output_image_path, result_image)

    # Optional: Display the original and modified images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Recolored Image")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    return output_image_path


# Predict skin and hair type
skin = predict_skin(image)
hairstyle = predict_hairtype(image)

if hairstyle.lower() == "curly" or hairstyle.lower() == "wavy":
    print(f"Hairstyle detected: {hairstyle}, proceeding with recoloring...")
    skin_color = (45, 98, 150)  # Brown shade (BGR)
    #(106, 178, 255)
    hair_color = (0, 0, 0)     # Black shade (BGR)

    # Call the recolor function for "shortwave.jpg"
    input_image_path = "shortwave.jpg"
    output_image_name = "shortwave_recolored.jpg"
    output_path = recolor_image(input_image_path, output_image_name, skin_color, hair_color)

    print(f"Recolored image saved at: {output_path}")
else:
    print(f"No recoloring applied. Hairstyle detected: {hairstyle}")

