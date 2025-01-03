import tensorflow as tf
import matplotlib.pyplot as plt


## Load the data
img_height, img_width = 32, 32
batch_size = 20
train_ds = tf.keras.utils.image_dataset_from_directory("fruits/train", image_size=(img_height, img_width), batch_size=batch_size)
valid_ds = tf.keras.utils.image_dataset_from_directory("fruits/validation", image_size=(img_height, img_width), batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory("fruits/test", image_size=(img_height, img_width), batch_size=batch_size)


## Visualize the data
class_names = ['apple', 'banana', 'orange']

plt.figure(figsize=(5, 5))

for image,label in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].numpy().astype("uint8"))
        plt.title(class_names[label[i]])
        plt.axis("off")
plt.show()


## Couche de prétraitement
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

## Create the model
modelCNN = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

## Compile the model
modelCNN.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

## Entrainement du model
history = modelCNN.fit(train_ds, validation_data=valid_ds, epochs=10)


## fit the model
modelCNN.fit(train_ds, validation_data=valid_ds, epochs=10)

## Evaluate the model
modelCNN.evaluate(test_ds, verbose=2)

## Display Courbe de loss et de precision
plt.figure(figsize=(8, 4))

## Precision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("Precision")

## Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='upper right')
plt.title("Loss")
plt.show()

## Save the model
modelCNN.save("fruit_model.h5")