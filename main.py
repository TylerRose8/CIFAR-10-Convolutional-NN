import os

import keras.activations
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
# from tensorflow import keras
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import layers
from tensorflow.keras import models

from numba import cuda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


def plot(history):
    # Plot the loss function
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history.history['loss'], 'r', label='train')
    ax.plot(history.history['val_loss'], 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.set_title('l1: ' + str(l1) + '    l2: ' + str(l2), fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.show()

    # Plot the accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history.history['accuracy'], 'r', label='train')
    ax.plot(history.history['val_accuracy'], 'b', label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.show()


epochs = 50
learning_rate = 0.0001

l1 = 64
l2 = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_x = x_train.astype('float32') / 255  # normalization
test_x = x_test.astype('float32') / 255
train_y = to_categorical(y_train)  # create label vectors
test_y = to_categorical(y_test)

best_model = None
best_model_history = None

def scheduler(epoch, lr):
    if lr < 0.0001:
        return 0.0001
    else:
        return lr

model = models.load_model('keras_CNN_CIFAR10_nonnotebook.model')
model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
time_start = time.time()
best_lr = None
best_acc = 0
while best_acc < 0.1:
    lr_callback = LearningRateScheduler(lambda epoch, lr: scheduler(epoch, lr))
    history = model.fit(train_x, train_y, batch_size=200, epochs=epochs, validation_data=(test_x, test_y),
                        callbacks=[callback, lr_callback], use_multiprocessing=True)
    # plot(history)
    acc = history.history['val_accuracy'][-1]
    if acc > best_acc:
        best_model = model
        best_model_history = history
        best_acc = acc
        best_lr = learning_rate
        model.save('keras_CNN_CIFAR10_EvenBetter.model')
    learning_rate /= 10
    # history = model.fit(train_x, train_y, batch_size=200, epochs=epochs,
    #                     validation_data=(test_x, test_y), verbose=1,
    #                     use_multiprocessing=True, callbacks=[callback])
    # model.hyperparameters_ = {'epochs': epochs, 'learning_rate': lr}
    # if history.history['val_accuracy'][-1] > best_acc:
    #     best_acc = history.history['val_accuracy'][-1]
    #     best_lr = lr
# model.save('keras_CNN_CIFAR10_best.model')
best_model.save('keras_CNN_CIFAR10_EvenBetter.model')

print("Training time: %s seconds" % (time.time() - time_start))

plot(history)

#
# for filterCount in range(1, 2+1):
#     for filterSize1 in range(1, 5+1, 2):
#         for filterSize2 in range(1, 5+1, 2):
#             for poolSize in range(1, 5+1, 2):
#                 for dense1Size in [64, 128, 256]:
#                     for dense2Size in [32, 64, 128]:
#                         model = models.Sequential()
#                         model.add(layers.Conv2D(32, (1, filterSize1), activation='relu', input_shape=(32, 32, 3)))
#                         model.add(layers.Conv2D(32, (filterSize1, 1), activation='relu'))
#                         if filterCount > 1:
#                             model.add(layers.Conv2D(32, (1, filterSize2), activation='relu'))
#                             model.add(layers.Conv2D(32, (filterSize2, 1), activation='relu'))
#                         model.add(layers.MaxPooling2D((poolSize, poolSize)))
#                         model.add(layers.Flatten())
#                         model.add(layers.Dense(dense1Size, activation='relu'))
#                         model.add(layers.Dense(dense2Size, activation='relu'))
#                         model.add(layers.Dense(10, activation='softmax'))
#
#                         callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)
#                         rmsprop = tf.keras.optimizers.RMSprop(
#                             learning_rate=learning_rate)
#                         model.compile(optimizer=rmsprop,
#                                       loss='categorical_crossentropy',
#                                       metrics=['accuracy'], )
#                         model.summary()
#                         time_start = time.time()
#
#                         history = model.fit(train_x, train_y, batch_size=200, epochs=epochs,
#                                             validation_data=(test_x, test_y), verbose=1,
#                                             use_multiprocessing=True, callbacks=[callback])
#                         print("Training time: %s seconds" % (time.time() - time_start))
#                         if best_model is None or best_model_history.history['val_accuracy'][-1] < history.history['val_accuracy'][-1]:
#                             best_model = model
#                             best_model_history = history
#                             model.save('keras_CNN_CIFAR10_nonnotebook.model')
#
# if best_model is not None:
#     best_model.save('keras_CNN_CIFAR10_nonnotebook.model')
#     best_model.summary()
#     plot(best_model_history)
