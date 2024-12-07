from transformers import pipeline
from PIL import Image

# Use Hugging Face pipeline for image classification
pipe = pipeline("image-classification", model="enzostvs/hair-color")

def classify_hair_color(image_path):
    """
    Classify the hair color of the given input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: The predicted hair color.
    """
    # Pass the image through the pipeline
    predictions = pipe(image_path)

    # Get the top class prediction
    predicted_label = predictions[0]['label']
    confidence = predictions[0]['score']

    return predicted_label

print(classify_hair_color('eddie_teter.jpeg'))
