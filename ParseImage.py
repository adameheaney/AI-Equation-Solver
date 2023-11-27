import cv2
import numpy as np

class ImageParser():

    def __init__(self, imagePath: str):
        self.imagePath = imagePath

    def displayImage(self, cvImage, name):
        cv2.imshow(name, cvImage)
        cv2.waitKey(0)

    def changeImagePath(self, imagePath: str):
        self.imagePath = imagePath

    #for testing the image parser
    def parseImageTesting(self) -> list:
        image = cv2.imread(self.imagePath)
        if image is None:
            print("NO IMAGE FOUND")
            exit()
        grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, grayscaleImage = cv2.threshold(grayscaleImage, 88, 255, cv2.THRESH_BINARY)
        
        # The following lines "fill" in the black parts (thanks chatgpt)
        # Set the size of the kernel for morphological operations
        kernel_size = 5
        # Create a rectangular kernel for opening operation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Apply morphological opening to remove white noise in black areas
        grayscaleImage = cv2.morphologyEx(grayscaleImage, cv2.MORPH_OPEN, kernel)

        #edge detection filter
        edges = cv2.Canny(grayscaleImage, 500, 1500)

        # Find contours in the edge-detected image. A contour is a mathematical representation of an image part's bounding
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        parts_of_equation = []

        # go through the contours to find the bounding box of them and then crop the image based on that
        for contour in contours:
            # Skips if the size of a contour is small (this effectively removes any false-objects)
            contour_size = cv2.contourArea(contour)
            if contour_size < 50:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cropped_part = grayscaleImage[y:y+h, x:x+w]
            # resize to 28 by 28 for the neural network
            cropped_part = cv2.resize(cropped_part, (28, 28))
            parts_of_equation.append(cropped_part)

        '''
        # Display images
        for part in parts_of_equation:
            self.displayImage(part, 'part')
        '''

        return parts_of_equation

def main():
    ip = ImageParser('images/handwrittenequation.jpg')
    ip.parseImageTesting()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation2.jpg')
    ip.parseImageTesting()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation3.jpg')
    ip.parseImageTesting()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation4.jpg')
    ip.parseImageTesting()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation5.jpg')
    ip.parseImageTesting()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    

