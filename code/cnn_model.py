from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.models import load_model

class CNNModel():
    def __init__(self, input_shape):
        model = Sequential()

        # Input layer
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # Subsequent conv blocks with spatial downsampling for downsampling
        model.add(Conv2D(128, (3, 3), padding='same', strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), padding='same', strides=2))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        # Dilated convolutions
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=1))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

        # Upscaling blocks
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

        # Output layer with tanh activation for predicting a and b channels
        model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

        return model

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, batch_size, epochs, validation_data):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def get_model(self):
        return self.model