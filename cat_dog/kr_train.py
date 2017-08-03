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
    vgg = VGG16(weights='imagenet', include_top=False, pooling='max')
    vgg.trainable = False

    inp = Input(shape=(224, 224, 3))
    x = BatchNormalization()(inp)
    x = vgg(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)

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

    print('# Cats in Train:', np.sum(y_train[:, 0]))
    print('# Dogs in Train:', np.sum(y_train[:, 1]))
    print('# Cats in Val:', np.sum(y_val[:, 0]))
    print('# Dogs in Val:', np.sum(y_val[:, 1]))

    name = 'keras_cnn'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    Path('log/').mkdir(exist_ok=True)
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
            # ModelCheckpoint(filepath=weight_path)
        ],
    } # yapf: disable
    model.fit(**fit_arg)


if __name__ == '__main__':
    main()