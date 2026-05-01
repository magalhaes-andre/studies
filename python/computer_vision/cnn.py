import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plot
from tabulate import tabulate

input_matrix = np.array([[1, 2, 3, 0, 1],
                        [4, 5, 6, 1, 2],
                        [7, 8, 9, 0, 1],
                        [0, 1, 2, 3, 4],
                        [1, 2, 3, 4, 5]])

print("INPUT MATRIX = \n", tabulate(input_matrix, tablefmt='grid'))

filter_matrix = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])

print("FILTER MATRIX = \n", tabulate(filter_matrix, tablefmt='grid'))

convolution_matrix = np.zeros((3, 3)) # starts an empty 3x3 matrix 

for i in range(3):
    for j in range(3):
        region = input_matrix[i:i+3, j:j+3]
        convolution_matrix[i, j] = np.sum(region * filter_matrix)

print("CONVOLUTION MATRIX = \n", tabulate(convolution_matrix, tablefmt='grid'))


# Loading MNIST dataset and preprocess.

(train_imgs, train_labels), (test_imgs, test_labels) = datasets.mnist.load_data()
train_imgs = train_imgs.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_imgs = test_imgs.reshape((10000, 28, 28, 1)).astype('float32') / 255

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Plotting the first 25 images of the training set

# Plotting the first 25 images of the training set

plot.figure(figsize=(10, 10))              # Create a new figure window with size 10x10 inches

for i in range(25):                        # Loop through the first 25 images
    plot.subplot(5, 5, i+1)                # Create a 5x5 grid of subplots, place image in position i+1
    plot.xticks([])                        # Remove x-axis tick marks
    plot.yticks([])                        # Remove y-axis tick marks
    plot.grid(False)                       # Turn off the grid lines
    plot.imshow(train_imgs[i], cmap=plot.cm.binary)  # Show the image in black-and-white (binary colormap)
    plot.xlabel(class_names[train_labels[i]])        # Label the image with its class name (digit)

plot.show()                                # Display the figure with all 25 images


# Fully Connected Layers

# First - Build the CNN
model = models.Sequential() # Create a sequential model (layers stacked in order)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # Add a convolutional layer with 32 filters, each 3x3, ReLU activation, input shape 28x28x1 (grayscale image)
model.add(layers.MaxPooling2D((2, 2))) # Add a max pooling layer to reduce spatial size (2x2 pooling)
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Add another convolutional layer with 64 filters, 3x3 size, ReLU activation
model.add(layers.MaxPooling2D((2, 2))) # Another max pooling layer (2x2)
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Third convolutional layer with 64 filters, 3x3 size, ReLU activation

# Then - Fully Connected Layers
model.add(layers.Flatten()) # Flatten the 2D feature maps into a 1D vector
model.add(layers.Dense(64, activation='relu')) # Fully connected (dense) layer with 64 neurons, ReLU activation
model.add(layers.Dense(10, activation='softmax')) # Output layer with 10 neurons (one per digit 0–9), softmax activation for probabilities
model.summary()

# Model Compilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(train_imgs, train_labels, epochs=5,
                    validation_data=(test_imgs, test_labels))


# Model Evaluation
test_loss, test_acc = model.evaluate(test_imgs, test_labels)
print(f'Test accuracy: {test_acc}')


# Predictions From the Compiled Model

predictions = model.predict(test_imgs)
plot.figure(figsize=(10, 10))
for i in range(25):
    plot.subplot(5, 5, i+1)
    plot.xticks([])
    plot.yticks([])
    plot.grid(False)
    plot.imshow(test_imgs[i].reshape(28,28), cmap=plot.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plot.xlabel(f'Predicted: {predicted_label} \n True: {true_label}', color=color)
plot.show()