import typer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def load_model():
    visible = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, kernel_size=3, activation='relu')(visible)
    drop1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D((2, 2))(drop1)
    flat1 = Flatten()(pool1)
    conv2 = Conv2D(32, kernel_size=3, activation='relu')(pool1)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D((2, 2))(drop2)
    flat2 = Flatten()(pool2)
    conv3 = Conv2D(32, kernel_size=3, activation='relu')(pool2)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D((2, 2))(drop3)
    flat3 = Flatten()(pool3)
    merge = tf.keras.layers.concatenate([flat1, flat2, flat3])
    hidden1 = Dense(units=100, activation='relu')(merge)
    output = Dense(units=10, activation='softmax')(hidden1)

    cnn = tf.keras.models.Model(inputs=visible, outputs=output)
    return cnn


if __name__ == '__main__':
    typer.run(load_model)