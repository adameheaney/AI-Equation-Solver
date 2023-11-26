from ast import main
import cv2
import numpy as np

class ImageParser():

    def __init__(self, imagePath: str):
        self.imagePath = imagePath

    def displayImage(self, cvImage):
        cv2.imshow('Image', cvImage)
        cv2.waitKey(0)

    def changeImagePath(self, imagePath: str):
        self.imagePath = imagePath

    #for testing the image parser
    def parseImageTesting(self):
        image = cv2.imread(self.imagePath)
        if image is None:
            print("NO IMAGE FOUND")
            exit()
        grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.displayImage(grayscaleImage)
        _, grayscaleImage = cv2.threshold(grayscaleImage, 85, 255, cv2.THRESH_BINARY)
        self.displayImage(grayscaleImage)

        #edge detection filter
        edges = cv2.Canny(grayscaleImage, 500, 1500)
        self.displayImage(edges)

        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours in the closed edges image
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over contours and extract individual objects
        for i, contour in enumerate(contours):
            
            # Skips if the size of a contour is small (this effectively removes any false-objects)
            contour_size = cv2.contourArea(contour)
            if contour_size < 50:
                continue
            
            # Create a mask for each object
            mask = np.zeros_like(edges)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            # Extract the object using the mask
            object_image = cv2.bitwise_and(edges, mask)

            # Save or display the individual object images
            cv2.imshow(f'Object {i+1}', object_image)
            cv2.waitKey(0)

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
    

