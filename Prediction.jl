py"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import  layers, models
from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def predict( test_images ,  weights_name = 'Weights_giga_B5.h5' ):

    conv_base = EfficientNetB0(weights=None, include_top=False, input_shape=(50, 50, 3))
    dropout_rate = 0.35
    reg = 0.0003
    model = models.Sequential()
    model.add(conv_base)
    # model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1024, activation="selu", kernel_regularizer=regularizers.l1(reg), activity_regularizer=regularizers.l1(reg) ) )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="selu",  kernel_regularizer=regularizers.l1(reg), activity_regularizer=regularizers.l1(reg) ) )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(256, activation="selu",  kernel_regularizer=regularizers.l1(reg), activity_regularizer=regularizers.l1(reg) ) )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(128, activation="selu",  kernel_regularizer=regularizers.l1(reg), activity_regularizer=regularizers.l1(reg) ) )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    model.load_weights(weights_name)

    eval_predict = model.predict(test_images)

    return np.array(eval_predict)

"""
