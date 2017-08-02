from pathlib import Path
from datetime import datetime

import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import *
from keras.optimizers import *
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, CSVLogger


def main():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(5, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(10, kernel_size=4, strides=2, activation='relu'))
    model.add(Conv2D(15, kernel_size=3, strides=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model_arg = {
        'loss': 'categorical_crossentropy',
        'optimizer': 'sgd',
        'metrics': ['accuracy']
    }
    model.compile(**model_arg)
    model.summary()

    train = np.load('train.npz')
    x_train, y_train = train['xs'], train['ys']
    val = np.load('val.npz')
    x_val, y_val = val['xs'], val['ys']

    name = 'kaggle_vgg'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = 'log/{} ({}).csv'.format(name, now)
    weight_path = '/tmp/' + name + '_{epoch:02d}_{val_acc:.3f}.h5'

    fit_arg = {
        'x': x_train,
        'y': y_train,
        'batch_size': 50,
        'epochs': 100,
        'shuffle': True,
        'validation_data': (x_val, y_val),
        'callbacks': [
            CSVLogger(log_path),
            ModelCheckpoint(filepath=weight_path)
        ],
    } # yapf: disable
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()