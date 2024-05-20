import keras
import torch
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

class CNNModel(keras.Model):
    def __init__(self, *args, **kwargs):

        # super(CNNModel, self).__init__(*args, **kwargs)
        self.build_model()

    def build_model(self):
        grey_image = keras.Input(shape=(256, 256, 1))

        model = keras.Sequential([
            keras.layers.Rescaling(1 / 255),

            # conv1
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(64, 3, padding='same', strides=2, activation='relu'),
            keras.layers.BatchNormalization(),

            # conv2
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(128, 3, padding='same', strides=2, activation='relu'),
            keras.layers.BatchNormalization(),

            # conv3
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(256, 3, padding='same', strides=2, activation='relu'),
            keras.layers.BatchNormalization(),

            # conv4
            keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            # conv5
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.BatchNormalization(),

            # conv6
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.Conv2D(512, 3, padding='same', dilation_rate=2, activation='relu'),
            keras.layers.BatchNormalization(),

            # conv7
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),

            # conv8
            keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu'),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
            keras.layers.BatchNormalization(),
        ])

        prediction = keras.layers.Conv2D(264, (1, 1), activation='softmax', padding='same', name='predictions')(model(grey_image))

        output = keras.layers.Conv2DTranspose(264, kernel_size=(3, 3), strides=(4, 4), padding='same', activation='softmax', name='output')(prediction)

        self.model = output

        super().__init__(inputs=grey_image, outputs=output)


    
    # def compile_model(self):
    #     self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == '__main__':

    model = CNNModel()
    model.summary()