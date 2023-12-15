from tabnanny import verbose
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from ParseImage import ImageParser
import ParseImage 

class EquationSolver():
 
    def __init__(self):
        #loads the model
        self.model = load_model('my_model.keras')

    def solve_equation(self, equation_path):
        # this string represents the equation
        equation_str = ""
        # parses through the equation image, creating a list of images of the parts of the equation
        parser = ImageParser(equation_path, 54)
        #for loop to predict each part of the equation. parser.parts_of_equation returns the list of images of the parts
        for img in parser.parts_of_equation:
            #converts the image into an array
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Make predictions
            predictions = self.model.predict(img_array, verbose=0)

            # Assuming you have 14 classes: these are ordered by how the folders are ordered in the dataset
            classes = ['/', '8', '5', '4', '-', '9', '1', '+', '7', '6', '3', '*', '2', '0']

            # Get the predicted class and add it to the string
            predicted_class_index = np.argmax(predictions)
            predicted_class = classes[predicted_class_index]
            equation_str += predicted_class

        #prints the equation and its result
        print('The parsed equation is ', equation_str)
        print('Your equation equals', str(eval(equation_str)))
def main():
    eqsolver = EquationSolver()
    #parser2 = ImageParser('images/handwrittenequation6.jpg', 54)
    #parser2.displayParts()
    eqsolver.solve_equation('images/handwrittenequation6.jpg')

if __name__ == '__main__':
    main()