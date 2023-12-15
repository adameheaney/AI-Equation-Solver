import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Set the input size for black and white images
input_size = (28, 28, 1)  # Grayscale images have a single channel

# Create the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))  # Optional dropout layer for regularization
model.add(layers.Dense(14, activation='softmax'))  # Adjusted for 14 classes

# Specify the learning rate (e.g., 0.001)
learning_rate = 0.001

# Create an instance of the Adam optimizer with your desired learning rate
custom_optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=custom_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the model summary
model.summary()

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

# Rescale test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Set the paths to your training and testing data
train_path = 'Data/Dataset/train'
test_path = 'Data/Dataset/eval'

# Create generators for training and testing data
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(28, 28),
    color_mode='grayscale',  # Specify grayscale
    batch_size=32,
    class_mode='categorical'  # Assuming one-hot encoding for classes
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,  # You can adjust the number of epochs
    validation_data=test_generator
)

# Save the model
model.save('my_model.keras')