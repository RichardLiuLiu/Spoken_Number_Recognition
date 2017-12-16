'''
Created on 2017.12.9

@author: Richard
'''
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

nrow = 200
ncol = 200
input_shape = (nrow, ncol, 3)
K.clear_session()

# Create a new model
model = Sequential()

model.add(Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()


train_data_dir = './spoken_numbers_wav/mfcc_image_tr'
batch_size_tr = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)
train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size_tr,
                        class_mode='sparse')

test_data_dir = './spoken_numbers_wav/mfcc_image_ts'
batch_size_ts = 5
test_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0,
                                   zoom_range=0,
                                   horizontal_flip=False)
test_generator = test_datagen.flow_from_directory(
                        test_data_dir,
                        target_size=(nrow,ncol),
                        batch_size=batch_size_ts,
                        class_mode='sparse')

model.compile(optimizer=optimizers.Adam(lr=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch =  train_generator.n // batch_size_tr
validation_steps =  test_generator.n // batch_size_ts

nepochs = 20  # Number of epochs

model.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = nepochs,
    validation_data = test_generator,
    validation_steps = validation_steps)

# Save the model
model.save('mfcc_cnn_model_all.h5')
