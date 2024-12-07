# turns the color output into hexcode and an actual image
from hairColor import classify_hair_color

from functions import predict_skin

from functions import predict_hairtype

print('uploaded')
image = 'eddie_teter.jpeg'
hair_color = classify_hair_color(image)

skin_color = predict_skin(image)

hair_type = predict_hairtype(image)

print(f"hair color : {hair_color}, hair style: {hair_type}, skin color: {skin_color}")