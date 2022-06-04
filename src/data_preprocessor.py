import typer
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('training input shape : ', X_train.shape)
    print('training output shape: ', y_train.shape)
    print('testing input shape  : ', X_test.shape)
    print('testing output shape : ', y_test.shape)

    return X_train, y_train, X_test, y_test


def preprocess(n_classes):
    X_train, y_train, X_test, y_test = load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    Y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    data = [X_train,Y_train,X_test,Y_test]

    print("Writing preprocessed data to file")
    for f in data:
        tf.io.write_file(
            f"data/{f}", f, name=None
        )


if __name__ == '__main__':
    typer.run(preprocess)
