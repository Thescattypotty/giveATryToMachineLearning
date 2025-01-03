from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## Load the VGG model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

## Extract features from the images
def extractFeatures(directory, augment=False):
    if augment:
        data_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        data_gen = ImageDataGenerator(rescale=1./255)
    
    generator = data_gen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    features = vgg_model.predict(generator)
    return features, generator.classes

## Extract features from the images
train_features, train_labels = extractFeatures('auto/train', augment=True)
test_features, test_labels = extractFeatures('auto/test')

## Create a model
model = Sequential()
model.add(Flatten(input_shape=train_features.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))  # 3 pour les marques

## Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_auto_model.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

## Train the model
model.fit(train_features, train_labels, epochs=50, batch_size=32, validation_data=(test_features, test_labels),
          callbacks=[early_stopping, model_checkpoint, reduce_lr])

loss, accuracy = model.evaluate(test_features, test_labels)

print(f'Loss: {loss}, Accuracy: {accuracy}')

## Save the model
model.save('CNN.h5')
