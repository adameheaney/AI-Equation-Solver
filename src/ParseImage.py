import cv2
import numpy as np

class ImageParser():

    parts_of_equation = []

    def __init__(self, imagePath: str, size):
        self.imagePath = imagePath
        self.parseImage(28)
        self.size = size

    def _displayImage(self, cvImage, name):
        cv2.imshow(name, cvImage)
        cv2.waitKey(0)
    def displayParts(self):
        for i, part in enumerate(self.parts_of_equation):
            cv2.imshow(f'part {i}', part)
        cv2.waitKey(0)
    def changeImagePath(self, imagePath: str):
        self.imagePath = imagePath
        self.parts_of_equation.clear()
        self.parseImage(self.size)

    #for testing the image parser
    def parseImage(self, size) -> list:
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
        self._displayImage(edges, 'name')
        

        # go through the contours to find the bounding box of them and then crop the image based on that
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # skips fragments
            if h * w < 350:
                continue
            cropped_part = grayscaleImage[y:y+h, x:x+w]
            # resize to size by size for the neural network
            padding_size = 6
            cropped_part = cv2.resize(cropped_part, (size - padding_size, size - padding_size))
            cropped_part = cv2.copyMakeBorder(cropped_part, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=255)
            self.parts_of_equation.append(cropped_part)

        '''
        # Display images
        for part in parts_of_equation:
            self._displayImage(part, 'part')
        '''

        return self.parts_of_equation

def main():
    ip = ImageParser('images/handwrittenequation.jpg', 28)
    ip.displayParts()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation2.jpg')
    ip.displayParts()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation3.jpg')
    ip.displayParts()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation4.jpg')
    ip.displayParts()
    cv2.destroyAllWindows()
    ip.changeImagePath('images/handwrittenequation5.jpg')
    ip.displayParts()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    

