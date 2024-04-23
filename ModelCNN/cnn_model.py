from keras.layers import Input, Conv2D, Model, BatchNormalization
from keras.models import load_model

from config import q_classes

class CNNModel():
    def __init__(self, image_shape=(256, 256, 1)):
        inputs = Input(shape=image_shape)

        x = Conv2D(64, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv1_1')(inputs)
        x = Conv2D(64, 3, activation='relu', padding='same', strides=2, dilation_rate=1, name='conv1_2')(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv2_1')(x)
        x = Conv2D(128, 3, activation='relu', padding='same', strides=2, dilation_rate=1, name='conv2_2')(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv3_1')(x)
        x = Conv2D(256, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv3_2')(x)
        x = Conv2D(256, 3, activation='relu', padding='same', strides=2, dilation_rate=1, name='conv3_3')(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv4_1')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv4_2')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv4_3')(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv5_1')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv5_2')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv5_3')(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv6_1')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv6_2')(x)
        x = Conv2D(512, 3, activation='relu', padding='same', strides=1, dilation_rate=2, name='conv6_3')(x)
        x = BatchNormalization()(x)


        x = Conv2D(256, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv7_1')(x)
        x = Conv2D(256, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv7_2')(x)
        x = Conv2D(256, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv7_3')(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, 3, activation='relu', padding='same', strides=.5, dilation_rate=1, name='conv8_1')(x)
        x = Conv2D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv8_2')(x)
        x = Conv2D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=1, name='conv8_3')(x)
        x = BatchNormalization()(x)

        outputs = Conv2D(q_classes, 1, activation='softmax', padding='same', name='output')(x)

        self.model = Model(inputs=inputs, outputs=outputs, name="ColorNet")

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
    
