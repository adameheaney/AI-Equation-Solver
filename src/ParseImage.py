import cv2
import numpy as np

class ImageParser():

    def __init__(self, imagePath: str, size):
        self.imagePath = imagePath
        self.size = size
        self.parts_of_equation= []
        self.parseImage(size)

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
    
    def _filter_overlapping_contours(self, contours):
        # Function to check if the bounding boxes of contour1 and contour2 overlap and contour1 is smaller
        def do_bounding_boxes_overlap_and_contour1_is_smaller(contour1, contour2):
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            # Check if bounding boxes overlap and contour1 is smaller
            return (
                x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                w1*h1 < w2*h2
            )

        # Iterate through contours and create a list of contours to keep
        filtered_contours = []
        for i, contour1 in enumerate(contours):
            is_contained = False
            for j, contour2 in enumerate(contours):
                if i != j and do_bounding_boxes_overlap_and_contour1_is_smaller(contour1, contour2):
                    is_contained = True
                    break
            if not is_contained:
                filtered_contours.append(contour1)

        return filtered_contours

    def parseImage(self, size) -> list:
        image = cv2.imread(self.imagePath)
        if image is None:
            print("NO IMAGE FOUND")
            exit()
        grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, grayscaleImage = cv2.threshold(grayscaleImage, 30, 255, cv2.THRESH_BINARY)
        
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
        #sort the contours
        contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
        # Iterate through contours and filter based on containment and size
        contours = self._filter_overlapping_contours(contours)
        #self._displayImage(edges, 'name')
    
        # go through the contours to find the bounding box of them and then crop the image based on that
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # skips fragments
            if h * w < 350:
                continue
            cropped_part = grayscaleImage[y:y+h, x:x+w]
            # resize to size by size for the neural network
            padding_size = max(w, h) - min(w, h)
            if padding_size % 2 != 0:
                padding_size - 1
            if max(w,h) == w:
                cropped_part = cv2.copyMakeBorder(cropped_part, padding_size//2, padding_size//2, 0, 0, cv2.BORDER_CONSTANT, value=255)
            else:
                cropped_part = cv2.copyMakeBorder(cropped_part, 0, 0, padding_size//2, padding_size//2, cv2.BORDER_CONSTANT, value=255)
            padding_size = 12
            cropped_part = cv2.resize(cropped_part, (size - padding_size, size - padding_size))
            cropped_part = cv2.copyMakeBorder(cropped_part,padding_size//2,padding_size//2,padding_size//2,padding_size//2, cv2.BORDER_CONSTANT, value=255)
            self.parts_of_equation.append(cropped_part)

        '''
        # Display images
        for part in parts_of_equation:
            self._displayImage(part, 'part')
        '''

        return self.parts_of_equation

def main():
    ip = ImageParser('images/handwrittenequation6.jpg', 54)
    ip.displayParts()
    cv2.destroyAllWindows()
    # ip.changeImagePath('images/handwrittenequation2.jpg')
    # ip.displayParts()
    # cv2.destroyAllWindows()
    # ip.changeImagePath('images/handwrittenequation3.jpg')
    # ip.displayParts()
    # cv2.destroyAllWindows()
    # ip.changeImagePath('images/handwrittenequation4.jpg')
    # ip.displayParts()
    # cv2.destroyAllWindows()
    # ip.changeImagePath('images/handwrittenequation5.jpg')
    # ip.displayParts()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    

