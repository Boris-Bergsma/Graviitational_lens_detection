py"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
from efficientnet.tfkeras import EfficientNetB5, EfficientNetB3 , EfficientNetB0
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc


import neural_structured_learning as nsl


from sklearn.metrics import roc_curve,roc_auc_score , precision_recall_fscore_support

import neural_structured_learning as nsl

import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

def training(train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):


    conv_base = EfficientNetB5(weights=None, include_top=False, input_shape=(50, 50, 3))

    dropout_rate = 0.35
    reg = 0.0001
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
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()


    # datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
    datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , rotation_range=30, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    datagen.fit(train_images)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, min_delta=0.0001)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=0.0001)
    model.compile(optimizer=Adam(lr=0.0003134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if save:
        model.load_weights('Weights_B5_new.h5')

    Batch_size = 256

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=(len(train_images[:,1,1,1])/Batch_size), callbacks = [callback], validation_data=(test_images, test_labels), verbose = 1)

    model.save_weights('Weights_B5_new.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("The accuracy is : ")
    print(test_acc)

    eval_predict = model.predict(test_images)
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(history.history['accuracy']) , np.array(history.history['val_accuracy']) , np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc




def training_rgb(  train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):


    inputs1 = tf.keras.Input(shape=(50, 50, 1))
    inputs2 = tf.keras.Input(shape=(50, 50, 1))
    inputs3 = tf.keras.Input(shape=(50, 50, 1))
    inputs_rgb = tf.keras.Input(shape=(50, 50, 3))

    conv_base_1 = EfficientNetB0(weights=None, include_top=False , input_shape=(50, 50, 1) )
    conv_base_1._name = "conv_base_r"
    x = conv_base_1(inputs1)
    Model_fat = tf.keras.layers.GlobalAvgPool2D()(x)


    conv_base_2 = EfficientNetB0(weights=None, include_top=False,  input_shape=(50, 50, 1) )
    conv_base_2._name = "conv_base_g"
    y = conv_base_2(inputs2)
    Model_medium = tf.keras.layers.GlobalAvgPool2D()(y)

    conv_base_3 = EfficientNetB0(weights=None, include_top=False ,  input_shape=(50, 50, 1) )
    conv_base_3._name = "conv_base_b"
    z = conv_base_3(inputs3)
    Model_tiny = tf.keras.layers.GlobalAvgPool2D()(z)

    conv_base_4 = EfficientNetB3(weights=None, include_top=False ,  input_shape=(50, 50, 3) )
    conv_base_4._name = "conv_base_rgb"
    rgb = conv_base_4(inputs_rgb)
    Model_rgb = tf.keras.layers.Flatten()(rgb)

    Merged = tf.keras.layers.concatenate([Model_fat, Model_medium,Model_tiny, Model_rgb])

    final = tf.keras.layers.Flatten()(Merged)
    final = tf.keras.layers.Dense(512, activation='selu', kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.0001) )(final)
    final = tf.keras.layers.Dropout(0.35)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(246, activation='selu', kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.0001) )(final)
    final = tf.keras.layers.Dropout(0.35)(final)
    final = tf.keras.layers.BatchNormalization()(final)

    output = tf.keras.layers.Dense(1,activation="sigmoid", kernel_regularizer=regularizers.l1(0.0005), activity_regularizer=regularizers.l1(0.0001) )(final)
    model = tf.keras.Model(inputs = [inputs1 , inputs2 , inputs3, inputs_rgb], outputs= output)

    # gen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    gen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)


    def generator_three_img(X1, X2, X3 , X4, y, batch_size):
        gc.collect()
        genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
        genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
        genX3 = gen.flow(X3, y, batch_size=batch_size, seed=1)
        genX4 = gen.flow(X4, y, batch_size=batch_size, seed=1)
        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            X4i = genX4.next()
            yield [X1i[0], X2i[0], X3i[0] , X4i[0]], X1i[1]

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, min_delta=0.0001)

    if save:
        model.load_weights('Weights_tripple_B5.h5')

    # conv_base_1.trainable = False
    # conv_base_2.trainable = False
    # conv_base_3.trainable = False

    model.compile(optimizer=Adam(lr=0.003134),loss = 'binary_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, min_delta=0.0001)

    model.summary()

    length_tr = len(train_images[:,1,1,1])
    length_te = len(test_images[:,1,1,1])

    Batch_size = 128

    history = model.fit_generator( generator_three_img(train_images[:,:,:,0].reshape((length_tr, 50, 50 , 1)) , train_images[:,:,:,1].reshape((length_tr, 50, 50 , 1)), train_images[:,:,:,2].reshape((length_tr, 50, 50 , 1)) , train_images  , train_labels, batch_size=Batch_size ),
     shuffle=True , epochs=EPOCHS, steps_per_epoch= np.floor(len(train_images[:,1,1,1])/Batch_size),
     validation_data=( [test_images[:,:,:,0].reshape((length_te, 50, 50 , 1)) , test_images[:,:,:,1].reshape((length_te, 50, 50 , 1)), test_images[:,:,:,2].reshape((length_te, 50, 50 , 1)), test_images], test_labels), callbacks = [callback], verbose = 1)
    #
    # history = model.fit( [train_images[:,:,:,0].reshape((length_tr, 50, 50 , 1)) , train_images[:,:,:,1].reshape((length_tr, 50, 50 , 1)), train_images[:,:,:,2].reshape((length_tr, 50, 50 , 1)) , train_images ] , train_labels, batch_size=128 ,
    #  shuffle=True , epochs=EPOCHS, callbacks = [callback],
    #  validation_data=( [test_images[:,:,:,0].reshape((length_te, 50, 50 , 1)) , test_images[:,:,:,1].reshape((length_te, 50, 50 , 1)), test_images[:,:,:,2].reshape((length_te, 50, 50 , 1)), test_images], test_labels), verbose = 1)

    model.trainable = True
    model.save_weights('Weights_tripple_B5.h5')

    test_loss, test_acc = model.evaluate([test_images[:,:,:,0].reshape((length_te, 50, 50 , 1)) , test_images[:,:,:,1].reshape((length_te, 50, 50 , 1)), test_images[:,:,:,2].reshape((length_te, 50, 50 , 1)), test_images], test_labels, verbose=2)
    print("The accuracy is : ")
    print(test_acc)

    eval_predict = model.predict([test_images[:,:,:,0].reshape((length_te, 50, 50 , 1)) , test_images[:,:,:,1].reshape((length_te, 50, 50 , 1)), test_images[:,:,:,2].reshape((length_te, 50, 50 , 1)), test_images])
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(history.history['accuracy']) , np.array(history.history['val_accuracy']) , np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc



def training_mono(train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):

    conv_base = EfficientNetB3(weights=None, include_top=False, input_shape=(50, 50, 1))

    dropout_rate = 0.1
    model = models.Sequential()
    model.add(conv_base)
    # model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1024, activation="selu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="selu", kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001) ) )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(256, activation="selu",  kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001) ) )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(128, activation="selu",  kernel_regularizer=regularizers.l1(0.001), activity_regularizer=regularizers.l1(0.001) ) )
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()


    # datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    # datagen.fit(train_images)

    callback = tf.keras.callbacks.EarlyStopping(monitor='v', patience=30, restore_best_weights=True, min_delta=0.0001)
    model.compile(optimizer=Adam(lr=0.00003134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if save:
        model.load_weights('Weights_B3_mono.h5')

    Batch_size = 256

    history = model.fit(datagen.flow(train_images.reshape((len(train_images[:,1,1]), 50, 50 , 1)), train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=np.floor(len(train_images[:,1,1,1])/Batch_size), callbacks = [callback], validation_data=(test_images.reshape((len(test_images[:,1,1]), 50, 50 , 1)), test_labels), verbose = 1)

    model.save_weights('Weights_B3_mono.h5')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)

    return np.array(history.history['accuracy']) , np.array(history.history['val_accuracy'])




def training_tripple_effnet(train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):

    inputs = tf.keras.Input(shape=(50, 50, 3))

    conv_base_1 = EfficientNetB0(weights="imagenet", include_top=False,  input_shape=(50, 50, 3) )
    conv_base_1._name = "conv_base_1"
    x = conv_base_1(inputs)
    Model_fat = tf.keras.layers.GlobalMaxPooling2D()(x)


    conv_base_2 = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(50, 50, 3) )
    conv_base_2._name = "conv_base_2"
    y = conv_base_2(inputs)
    Model_medium = tf.keras.layers.GlobalMaxPooling2D()(y)

    conv_base_3 = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(50, 50, 3) )
    conv_base_3._name = "conv_base_3"
    z = conv_base_3(inputs)
    Model_tiny = tf.keras.layers.GlobalMaxPooling2D()(z)

    Merged = tf.keras.layers.concatenate([Model_fat, Model_medium, Model_tiny])

    final = tf.keras.layers.Flatten()(Merged)
    final = tf.keras.layers.Dense(1024, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.3)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(1024, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.3)(final)
    final = tf.keras.layers.BatchNormalization()(final)
    final = tf.keras.layers.Dense(1024, activation='relu')(final)
    final = tf.keras.layers.Dropout(0.3)(final)
    final = tf.keras.layers.BatchNormalization()(final)

    outputs = tf.keras.layers.Dense(1,activation="sigmoid")(final)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=Adam(lr=0.0003143),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    datagen.fit(train_images)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=0.0001)
    model.compile(optimizer=Adam(lr=0.0003134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if save:
        model.load_weights('Tripple_Eff_net_backup.h5')

    Batch_size = 256

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=(len(train_images[:,1,1,1])/Batch_size), callbacks = [callback], validation_data=(test_images, test_labels), verbose = 1)

    model.save_weights('Tripple_Eff_net_tt.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("The accuracy is : ")
    print(test_acc)

    eval_predict = model.predict(test_images)
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(history.history['accuracy']) , np.array(history.history['val_accuracy']) , np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc


def training_GEN(train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):


    conv_base = EfficientNetB0(weights=None, include_top=False, input_shape=(50, 50, 3))

    dropout_rate = 0.35
    reg = 0.00001
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="selu",  kernel_regularizer=regularizers.l1(reg), activity_regularizer=regularizers.l1(reg) ) )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()


    datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , rotation_range=30, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    datagen.fit(train_images)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, min_delta=0.0001)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.0001)
    model.compile(optimizer=Adam(lr=0.0005134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if save:
        model.load_weights('Weights_B0_GEN.h5')

    Batch_size = 256

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=(len(train_images[:,1,1,1])/Batch_size), callbacks = [callback],  validation_data=(test_images, test_labels), verbose = 1)

    model.save_weights('Weights_B0_GEN.h5')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    eval_predict = model.predict(test_images)
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc


def training_scan(train_images, train_labels , test_images , test_labels, test_images_ratio, test_labels_ratio  ,candidates , EPOCHS = 10 , save = False):


    conv_base = EfficientNetB0(weights=None, include_top=False, input_shape=(50, 50, 3))

    dropout_rate = 0.2
    model = models.Sequential()
    model.add(conv_base)
    #model.add(layers.GlobalAvgPool2D())
    model.add(layers.Dense(1024, activation="selu" ) )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="selu") )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(256, activation="selu") )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(layers.Dense(128, activation="selu" ) )
    model.add(layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid", name="fc_out"))

    datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] , rotation_range=10, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , rotation_range=30, width_shift_range=0.1,height_shift_range=0.1)
    # datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , zoom_range = [1,1.2] )
    # datagen.fit(train_images)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, min_delta=0.0001)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.0001)
    model.compile(optimizer=Adam(lr=0.0003134),loss = 'binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if save:
        model.load_weights('Weights_B0_scan.h5')

    Batch_size = 256

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=(len(train_images[:,1,1,1])/Batch_size), validation_data=(test_images, test_labels), verbose = 1)

    model.save_weights('Weights_B0_scan.h5')

    test_loss, test_acc = model.evaluate(test_images_ratio, test_labels_ratio, verbose=2)
    eval_predict = model.predict(test_images_ratio)
    auc_score=roc_auc_score(test_labels_ratio,eval_predict)
    pre , rec , _ , _ = precision_recall_fscore_support(test_labels_ratio, 1*(eval_predict > 0.49) , average='macro')

    eval_predict_cand = model.predict(candidates)
    number_cand = np.mean(1*(eval_predict_cand > 0.49))

    return    auc_score , test_acc , pre , rec , number_cand



def adv_training( train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):



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

    # model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])

    # datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , rotation_range=30, width_shift_range=0.1,height_shift_range=0.1)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=50, restore_best_weights=True, min_delta=0.0001)


    adv_config = nsl.configs.make_adv_reg_config(multiplier=0.1 , adv_step_size= 0.001 , clip_value_min = 0, clip_value_max = 1, adv_grad_norm='infinity') # To have inf norm , adv_grad_norm='infinity'
    adv_model = nsl.keras.AdversarialRegularization(model,label_keys=['label'], adv_config=adv_config)

    adv_model.compile(optimizer=Adam(lr=0.0003134),loss = 'binary_crossentropy', metrics=['accuracy'])


    Batch_size = 256

    if save:
        adv_model.load_weights('Weights_B0_adv.h5')

    train_data = tf.data.Dataset.from_generator(
        datagen.flow_from_directory,
        output_types=(tf.float32, tf.float32),
        output_shapes = ([Batch_size,50,50,3],[Batch_size,1]))
    train_data = tf.data.Dataset.from_tensor_slices(
    {'input': train_images, 'label': train_labels}).batch(Batch_size)

    val_data = tf.data.Dataset.from_tensor_slices(
    {'input': test_images, 'label': test_labels}).batch(Batch_size)
    val_steps = test_images.shape[0] / Batch_size


    history =  adv_model.fit(train_data, validation_data=val_data, validation_steps=val_steps, epochs=EPOCHS, callbacks = [callback],  verbose=1) # callbacks = [callback]

    model.save_weights('Weights_B0_adv.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("The accuracy is : ")
    print(test_acc)

    eval_predict = model.predict(test_images)
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(history.history['binary_accuracy']) , np.array(history.history['val_binary_accuracy']) , np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc



def training_B0( train_images, train_labels , test_images , test_labels , EPOCHS = 10 , save = False):


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

    # model.compile(optimizer=Adam(lr=0.0000134),loss = 'binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=0.0003134),loss = 'binary_crossentropy', metrics=['accuracy'])


    # datagen = ImageDataGenerator(horizontal_flip=True , vertical_flip=True )
    datagen = ImageDataGenerator( horizontal_flip=True , vertical_flip=True , rotation_range=30, width_shift_range=0.1,height_shift_range=0.1)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, min_delta=0.0001)

    Batch_size = 256

    if save:
        model.load_weights('Weights_B0_test.h5')


    history = model.fit(datagen.flow(train_images, train_labels, batch_size=Batch_size, shuffle=True) , epochs=EPOCHS, steps_per_epoch=(len(train_images[:,1,1,1])/Batch_size), callbacks = [callback], validation_data=(test_images, test_labels), verbose = 1)


    model.save_weights('Weights_B0_test.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("The accuracy is : ")
    print(test_acc)

    eval_predict = model.predict(test_images)
    fpr , tpr , thresholds = roc_curve ( test_labels , eval_predict)

    auc_score=roc_auc_score(test_labels,eval_predict)

    print('AUC score: ')
    print(auc_score)

    return np.array(history.history['accuracy']) , np.array(history.history['val_accuracy']) , np.array(eval_predict), np.array(fpr), np.array(tpr) , auc_score , test_acc


"""
