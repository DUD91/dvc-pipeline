import typer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle
from datetime import datetime

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


def train(n_epochs,batch_size):
    model = load_model()


    X_train = tf.io.read_file("data/X_train", name=None)
    Y_train = tf.io.read_file("data/Y_train", name=None)
    X_test = tf.io.read_file("data/X_test", name=None)
    Y_test = tf.io.read_file("data/Y_test", name=None)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    log = model.fit(X_train, Y_train,
                    batch_size = batch_size,
                    epochs=n_epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))

    with open('data/train_history', 'wb') as file_pi:
        pickle.dump(log.history, file_pi)

    print("Saving Model")
    model_name = f'custom_cnn_{datetime.now()}'
    model.save(f'models/custom_cnn')

if __name__ == '__main__':
    typer.run(train)