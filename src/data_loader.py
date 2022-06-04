import typer
from tensorflow.keras.datasets import cifar10

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('training input shape : ', X_train.shape)
    print('training output shape: ', y_train.shape)
    print('testing input shape  : ', X_test.shape)
    print('testing output shape : ', y_test.shape)
    return (X_train, y_train), (X_test,y_test)


if __name__ == '__main__':
    (X_train, y_train), (X_test,y_test) = typer.run(load_data)
