import tensorflow as tf  # Importe la bibliothèque TensorFlow

import numpy as np       # Importe NumPy pour les tableaux et calculs numériques
import matplotlib.pyplot as plt  # Importe la bibliothèque de visualisation matplotlib

fashion_mnist = tf.keras.datasets.fashion_mnist  # Récupère le dataset Fashion MNIST

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # Charge les données d’entraînement et de test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Définit les noms des classes

train_images.shape  # Vérifie la dimension du tableau d’images d’entraînement

len(train_labels)   # Nombre d’étiquettes d’entraînement

train_labels        # Affiche la liste (ou le tableau) d’étiquettes

test_images.shape   # Vérifie la dimension du tableau d’images de test

len(test_labels)    # Nombre d’étiquettes de test

plt.figure()        # Crée une nouvelle figure pour la visualisation
plt.imshow(train_images[0])  # Affiche la première image d’entraînement
plt.colorbar()      # Ajoute une barre de couleurs
plt.grid(False)     # Retire la grille
plt.show()          # Affiche la figure

train_images = train_images / 255.0  # Normalise les images d’entraînement

test_images = test_images / 255.0    # Normalise les images de test

plt.figure(figsize=(10,10))  # Définit la taille de la figure
for i in range(25):          # Boucle pour afficher 25 images
    plt.subplot(5,5,i+1)     # Crée des sous-graphiques 5x5
    plt.xticks([])          
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # Affiche l’image en nuances de gris
    plt.xlabel(class_names[train_labels[i]])         # Affiche le nom de la classe
plt.show()        # Affiche les 25 images

model = tf.keras.Sequential([  # Crée un modèle séquentiel Keras
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Aplatissement des images 28x28
    tf.keras.layers.Dense(128, activation='relu'),  # Couche dense avec fonction ReLU
    tf.keras.layers.Dense(10)                       # Couche de sortie pour 10 classes
])

model.compile(optimizer='adam',  # Choix de l’optimiseur Adam
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])  # Indique la métrique d’exactitude

model.fit(train_images, train_labels, epochs=10)  # Entraîne le modèle sur 10 époques

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)  # Évalue sur le jeu de test

print('\nTest accuracy:', test_acc)  # Affiche l’exactitude en test

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])  # Ajoute une couche Softmax pour obtenir des probabilités


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])  # Même modèle de probabilité (redondant)

predictions = probability_model.predict(test_images)  # Prédit les sorties pour les images de test
predictions[0]  # Affiche les probabilités pour la première image

np.argmax(predictions[0])  # Récupère l’indice de la classe prédite

test_labels[0]             # Valeur réelle de la première étiquette


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]     # Sélectionne l’étiquette et l’image
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)         # Affiche l’image en nuances de gris

  predicted_label = np.argmax(predictions_array)  # Classe prédite
  if predicted_label == true_label:
    color = 'blue'      # Bleu si prédiction correcte
  else:
    color = 'red'       # Rouge si prédiction incorrecte

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)  # Affiche label, pourcentage, label réel

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]  # Étiquette réelle
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)  # Classe prédite

  thisplot[predicted_label].set_color('red')  # Barre rouge pour la prédiction
  thisplot[true_label].set_color('blue')      # Barre bleue pour la classe réelle
  
  
i = 0
plt.figure(figsize=(6,3))              # Crée une figure de taille 6x3
plt.subplot(1,2,1)                     # Placement du premier subplot
plot_image(i, predictions[i], test_labels, test_images)  # Affiche l’image et sa prédiction
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)        # Affiche le graphique en barres
plt.show()           # Montre la figure

i = 12
plt.figure(figsize=(6,3))             # Crée une figure de taille 6x3
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()          # Montre la figure


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5    # Nombre de lignes de figures
num_cols = 3    # Nombre de colonnes de figures
num_images = num_rows*num_cols  # Total d’images à afficher
plt.figure(figsize=(2*2*num_cols, 2*num_rows))  # Définit la taille globale
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()   # Affiche toutes les figures organisées

# Grab an image from the test dataset.
img = test_images[1]    # Sélectionne la deuxième image du test

print(img.shape)        # Affiche la forme du tableau

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))  # Ajoute une dimension pour en faire un batch de 1

print(img.shape)


predictions_single = probability_model.predict(img)  # Fait une prédiction pour cette image

print(predictions_single)


plot_value_array(1, predictions_single[0], test_labels)  # Affiche le graphique en barres pour la prédiction
_ = plt.xticks(range(10), class_names, rotation=45)      # Affiche les noms de classes inclinés
plt.show()


np.argmax(predictions_single[0])  # Indice de la classe prédite pour cette image


# Save the model to a file
model.save('ANN.h5')