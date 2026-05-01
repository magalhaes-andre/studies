import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

MID_POINT_OF_GRAYSCALE_RANGE = 127.5

# Load and pre-processing
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = (train_images.astype('float32') - MID_POINT_OF_GRAYSCALE_RANGE) / MID_POINT_OF_GRAYSCALE_RANGE
train_images = np.expand_dims(train_images, axis = -1)

def make_generator_model():
    """
    Build the GAN generator model.

    Architecture:
    - Dense(7×7×256): expands 100‑dimensional noise into a 7×7×256 feature map (seed image).
    - BatchNormalization + LeakyReLU: stabilizes training and ensures smooth gradient flow.
    - Reshape(7,7,256): converts dense output into a low‑resolution image grid.
    - Conv2DTranspose(128, stride=1): enriches features without changing resolution.
    - BatchNormalization + LeakyReLU: stabilizes and refines features.
    - Conv2DTranspose(64, stride=2): upsamples image from 7×7 → 14×14, reducing channels.
    - BatchNormalization + LeakyReLU: stabilizes and refines again.
    - Conv2DTranspose(1, stride=2, activation='tanh'): final upsampling to 28×28 grayscale,
      with outputs in [-1,1] to match normalized inputs.
    """
    model = models.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100, )))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model    

def make_discriminator_model():
    """
    Build the GAN discriminator model.

    Architecture:
    - Conv2D(64 filters, stride=2): extracts low‑level features, downsamples 28×28 → 14×14.
    - LeakyReLU: nonlinear activation, avoids dying ReLU problem.
    - Dropout(0.3): regularization, prevents overfitting.
    - Conv2D(128 filters, stride=2): deeper feature extraction, downsamples 14×14 → 7×7.
    - LeakyReLU + Dropout(0.3): stability and regularization.
    - Flatten(): converts 7×7×128 feature map into 1D vector.
    - Dense(1): outputs a single logit (real vs. fake confidence).
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training Definition
EPOCHS = 100
BATCH_SIZE = 256
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

generator = make_generator_model()
discriminator = make_discriminator_model()

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)

        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * MID_POINT_OF_GRAYSCALE_RANGE + MID_POINT_OF_GRAYSCALE_RANGE, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train_and_generate_images(dataset, epochs):
    for epoch in range(epochs):
        train(dataset, 1)
        generate_and_save_images(generator, epoch + 1, seed)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(len(train_images)).batch(BATCH_SIZE)
train_and_generate_images(train_dataset, EPOCHS)