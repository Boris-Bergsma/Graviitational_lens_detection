py"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# from tensorflow.keras.mixed_precision import experimental as mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
#
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)
# tf.enable_v2_behavior()

def image_preprocess(x):
  x['image'] = tf.cast(x['image'], tf.float32)
  return (x['image'],)  # (input, output) of the model

def train_pixel_cnn(train_images, train_labels , test_images , epochs = 30,  checkpoint = False):


    train_data = tf.data.Dataset.from_tensor_slices((train_images))
    test_data = tf.data.Dataset.from_tensor_slices((test_images))


    batch_size = 32
    train_it = train_data.batch(batch_size)

    # Define a Pixel CNN network
    dist = tfd.PixelCNN(
        image_shape=(50, 50, 3),
        num_resnet=3,
        num_hierarchies=2,
        num_filters=45,
        num_logistic_mix=5,
        dropout_p=.4,
        use_data_init=True, high=250, low=0,
    )

    # Define the model input
    image_input = tfkl.Input(shape=(50,50,3))

    # Define the log likelihood for the loss fn
    log_prob = dist.log_prob(image_input)

    # Define the model
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    # Compile and train the model
    model.compile(
    optimizer=tfk.optimizers.Adam(.0000714),
    metrics=[])

    if checkpoint :
        model.load_weights('PixelCNN.h5')

    datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)

    history = model.fit(datagen.flow(train_images, train_labels, batch_size = 16,  shuffle=True) , epochs=epochs, steps_per_epoch=(len(train_images[:,1,1,1])/16)) #, steps_per_epoch=(len(train_it[:,1,1,1])/Batch_size)

    model.save_weights('PixelCNN.h5')

    # sample five images from the trained model
    samples = dist.sample(16)

    return np.array(samples), np.array(history.history['loss'])

def train_pixel_cnn_CFIS(train_images, train_labels , test_images , epochs = 30,  checkpoint = False):

    train_images = np.reshape(train_images, (len(train_images[:,1,1]) , 44, 44, 1))

    train_data = tf.data.Dataset.from_tensor_slices((train_images))
    test_data = tf.data.Dataset.from_tensor_slices((test_images))


    batch_size = 32
    train_it = train_data.batch(batch_size)

    # Define a Pixel CNN network
    dist = tfd.PixelCNN(
        image_shape=(44, 44, 1),
        num_resnet=2,
        num_hierarchies=2,
        num_filters=30,
        num_logistic_mix=5,
        dropout_p=.2,
        use_data_init=True, high=250, low=0,
    )

    # Define the model input
    image_input = tfkl.Input(shape=(44,44,1))

    # Define the log likelihood for the loss fn
    log_prob = dist.log_prob(image_input)

    # Define the model
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    # Compile and train the model
    model.compile(
    optimizer=tfk.optimizers.Adam(.0001),
    metrics=[])

    if checkpoint :
        model.load_weights('PixelCNN_CFIS.h5')

    datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)

    history = model.fit(datagen.flow(train_images, train_labels, batch_size = 16,  shuffle=True) , epochs=epochs, steps_per_epoch=(len(train_images[:,1,1])/16)) #, steps_per_epoch=(len(train_it[:,1,1,1])/Batch_size)

    model.save_weights('PixelCNN_CFIS.h5')

    # sample five images from the trained model
    samples = dist.sample(16)

    return np.array(samples), np.array(history.history['loss'])



def predict_pixel_cnn(N_predictions = 17, batch_size = 16, weights = 'PixelCNN.h5'):

    # dist = tfd.PixelCNN(
    #     image_shape=(50, 50, 3),
    #     num_resnet=2,
    #     num_hierarchies=2,
    #     num_filters=30,
    #     num_logistic_mix=5,
    #     dropout_p=.2,
    #     use_data_init=True, high=250, low=0,
    # )
    dist = tfd.PixelCNN(
        image_shape=(50, 50, 3),
        num_resnet=3,
        num_hierarchies=2,
        num_filters=45,
        num_logistic_mix=5,
        dropout_p=.4,
        use_data_init=True, high=250, low=0,
    )

    # Define the model input
    image_input = tfkl.Input(shape=(50,50,3))

    # Define the log likelihood for the loss fn
    log_prob = dist.log_prob(image_input)

    # Define the model
    model = tfk.Model(inputs=image_input, outputs=log_prob)
    model.add_loss(-tf.reduce_mean(log_prob))

    # Compile and train the model
    model.compile(
        optimizer=tfk.optimizers.Adam(.001),
        metrics=[])


    model.load_weights(weights)

    samples = []

    for i in range(int(np.floor(N_predictions/batch_size))):
        samples.append(dist.sample(batch_size))
        print(i*batch_size)

    return np.array(samples)



def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*504, use_bias=False, input_shape=(1000,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((5, 5, 504)))
    assert model.output_shape == (None, 5, 5, 504) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(252, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 5, 5, 252)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(126, (3, 3), strides=(5, 5), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 126)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(63, (7, 7), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 25, 63)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(21, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 50, 21)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 100, 3)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(3, (2, 2), strides=(2, 2), padding='same', activation='selu'))
    assert model.output_shape == (None, 50, 50, 3)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=[50, 50, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (6, 6), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
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



def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] )
      plt.axis('off')

  plt.savefig('Train_steps_images/image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()




def test_train_julia(train_images, train_labels):


    train_images = train_images.reshape(train_images.shape[0], 50, 50, 3).astype('float32')

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


    generator = make_generator_model()
    generator.summary()

    noise = tf.random.normal([1, 1000])
    generated_image = generator(noise, training=False)

    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')


    discriminator = make_discriminator_model()
    decision = discriminator(generated_image)
    print (decision)
    discriminator.summary()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 159
    noise_dim = 1000
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])


    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = generator(noise, training=True)

          real_output = discriminator(images, training=True)
          fake_output = discriminator(generated_images, training=True)

          gen_loss = generator_loss(fake_output)
          disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    def train(dataset, epochs):
      for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
          train_step(image_batch)

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)

        # Save the model every 15 epochs

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

      # Generate after the final epoch
      generate_and_save_images(generator,
                               epochs,
                               seed)

    train(train_dataset, EPOCHS)


    generator.save_weights('generator.h5')
    discriminator.save_weights('CNN.h5')
"""
