# AI-Equation-Solver
This project is an AI that can solve basic handwritten equations.

# Method
Here is the essential outline of how the program will work: First, the program will separate the original image into multiple images of each piece of the equation, adding filters and resizing the parts to get them ready to be evaluated by the CNN. Next, the CNN will identify each image, and return the pieces in the order they were originally in. Finally, the equation will be solved, with its solution being returned. For parsing the image, the cv2 library (open computer vision) is taken advantage of. The methods used for parsing the images are as follows: Firstly, the image is converted into a grayscale image, and then subsequently thresholded. A morphological opening algorithm provided by cv2 is used to fill in any holes within the thresholded image. Next, a canny edge detection filter is applied to “scan” for the pieces within the equation, and contours are made of the edges. Next, for each contour, a new image is cropped out of the original image at the rectangular borders of the contour. This separates each piece into its image. Finally, the images are resized to the correct size for the CNN. The CNN was made using Tensorflow’s easy-to-build neural network library. The CNN is made up of 6 layers in total: an input layer, 3 convolutional layers each with pooling layers directly after, one fully connected layer, and finally the output layer. The input layer takes in 54x54 parameters since the images are 54x54 pixels.

# Result
The model accurately identified the image 95% of the time.

# Dataset
There are a total of 14 classes: \[0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, +, -, *(represented by an "x" symbol), /]
