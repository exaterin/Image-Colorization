from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.models import load_model

class CNNModel():
    def __init__(self, image_shape=(256, 256, 1)):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=image_shape))

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