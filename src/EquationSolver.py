import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from ParseImage import ImageParser

# Load the saved model
model = load_model('my_model.keras')

# equation image path
equation_path = 'images/handwrittenequation.jpg' 

# list of equation parts
equation = []

# test images
test_imgs = ["Data/Dataset/eval/plus val/127.jpg", 
             'Data/Dataset/eval/plus val/267.jpg',
             'Data/Dataset/eval/plus val/536.jpg',
             'Data/Dataset/eval/plus val/1193.jpg',
             'Data/Dataset/eval/plus val/1896.jpg']

# parse through the equation image
parser = ImageParser(equation_path, 28)
parser.displayParts()

for img in parser.parts_of_equation:
    #img = image.load_img(img, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)

    # Assuming you have 14 classes
    classes = ['/', '8', '5', '4', '-', '9', '1', '+', '7', '6', '3', '*', '2', '0']

    # Get the predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class = classes[predicted_class_index]
    equation = predicted_class

    print(f'The predicted class is: {predicted_class}')
