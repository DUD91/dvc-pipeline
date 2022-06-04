import typer

def train(model,X_train, Y_train, X_test, Y_test, n_epochs,batch_size):
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    log = model.fit(X_train, Y_train,
                    batch_size = batch_size,
                    epochs=n_epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
    return log

if __name__ == '__main__':
    typer.run(train)