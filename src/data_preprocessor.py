import typer
import tensorflow as tf

def preprocess(X_train,y_train, X_test,y_test,n_classes):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    Y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    Y_test = tf.keras.utils.to_categorical(y_test, n_classes)
    return (X_train,Y_train), (X_test,Y_test)



if __name__ == '__main__':
    typer.run(preprocess)