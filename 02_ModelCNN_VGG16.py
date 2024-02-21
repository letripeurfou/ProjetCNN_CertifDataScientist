#*---------------------------------------------------------------*#
#*------------------- CNN PRE-TRAINED - VGG16 -------------------*#
#*---------------------------------------------------------------*#

#!Preambule! Importation des librairies
import os
#import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import matplotlib.pyplot as plt
#from PIL import Image

import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


#*-------------------------------------------------------------*#
#*------------------ PREPARATION DES DONNEES ------------------*#
#*-------------------------------------------------------------*#

#*-------- Importation des dataframes --------*#
data_dir = 'data_input/garbage_classification_6classes'

# Ensemble d'entraînement
train_path = os.path.join(data_dir, 'train_df.csv')
train_data = pd.read_csv(train_path)

# Ensemble de validation
valid_path = os.path.join(data_dir, 'valid_df.csv')
valid_data = pd.read_csv(valid_path)

# Ensemble de test
test_path = os.path.join(data_dir, 'test_df.csv')
test_data = pd.read_csv(test_path)

#*----------------------------------------------------------*#
#*--------------- PRE-TRAITEMENT DES DONNEES ---------------*#
#*----------------------------------------------------------*#

#*-------- ENSEMBLE D'ENTRAINEMENT --------*#
# Création d'un générateur de données pour l'ensemble d'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Rééchelle les valeurs des pixels entre 0 et 1
    ### Augmentation des données ###
    rotation_range=20,          # Rotation aléatoire des images jusqu'à 40 degrés
    width_shift_range=0.2,      # Déplacement horizontal aléatoire de l'image
    height_shift_range=0.2,     # Déplacement vertical aléatoire de l'image
    horizontal_flip=True,       # Retournement horizontal aléatoire de l'image
    fill_mode="nearest"         # Mode de remplissage pour les transformations géométriques
    )

# Création d'un générateur d'images à partir du DataFrame d'entraînement
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,      # DataFrame contenant les informations sur les images et leurs classes 
    x_col='path',                # Colonne contenant les chemins des images
    y_col='class_',              # Colonne contenant les classes des images
    batch_size=16,               # Taille du lot d'images généré à chaque itération
    target_size=(224, 224),      # Taille cible des images après redimensionnement
    class_mode='categorical'     # Mode de classification pour les classes multiples
    )

#*-------- ENSEMBLE DE VALIDATION --------*#
# Mise à l'échelle des valeurs des images en appliquant le coeff 1/225
#!!!Remarque!!! Les données de l'ensemble de validation ne doivent pas être augmentées
valid_datagen = ImageDataGenerator(rescale=1./255)

# Création du générateur d'images 'validationGenerator' 
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_data,
    x_col='path',
    y_col='class_',
    batch_size=16,
    target_size=(224, 224),
    class_mode="categorical"
    )

#*-----------------------------------------------------------*#
#*--------------- MODELE PRE-ENTRAINE - VGG16 ---------------*#
#*-----------------------------------------------------------*#

# définition du proxy
# A commenter si non besoin
proxy_host = 'http://vip-users.proxy.edf.fr'
proxy_port = '3131'
os.environ['http_proxy'] = f"{proxy_host}:{proxy_port}"
os.environ['https_proxy'] = f"{proxy_host}:{proxy_port}"


# Instanciation de la base de convolution d'un modèle VGG16
from keras.applications import VGG16
model_vgg16 = VGG16(weights='imagenet',  #~ arg 'weight' spécifie le weight checkpoint à partir duquel le modèle est initialisé
                  include_top=False,  #~ 'include_top' réfère à l'inclusion (ou non) du classifieur entièrement connecté sur la partie supérieur du réseau (par défaut 1 000 classes d'ImageNet) ; usage ici de notre propre classifieur
                  input_shape=(224, 224, 3))  #~ 'input_shape' renseigne la forme des tenseurs d'images envoyés au réseau

model_vgg16.summary()

#*--------------------------------------------------*#
#*------------- REGLAGES DE PRECISION --------------*#
#*--------------------------------------------------*#

# !Remarque! Seules les couches du dernier block (block5) soit :
# block5_conv1 / block5_conv2 / block5_conv3 doivent pouvoir être entraînés 

print('Nombre de poids entraînables avant de geler la base convolutive:',
      len(model_vgg16.trainable_weights)) 

# Création variable booléenne : indique si les couches doivent être rendues entraînables
set_trainable = False

# Parcours de chaque couche du modèle VGG16
for layer in model_vgg16.layers:
    # Condition IF : vérifie si le nom de la couche est 'block5_conv1'
    if layer.name == 'block5_conv1':
        # Si oui : active le flag pour rendre les couches suivantes entraînables
        set_trainable = True
    layer.trainable = set_trainable

print('Nombre de poids entraînables pour le dernier block5:',
      len(model_vgg16.trainable_weights)) # 6 

for layer in model_vgg16.layers:
    if layer.trainable:
        print("Layer '{}' is trainable".format(layer.name))
    else:
        print("Layer '{}' is not trainable".format(layer.name))


#*----------------------------------------------------*#
#*-------------- ARCHITECTURE DU RESEAU --------------*#
#*----------------------------------------------------*#

model_final_vgg16 = models.Sequential()

# Ajout du modele VGG16
model_final_vgg16.add(model_vgg16)

# Ajout d'un classifieur entièrement connecté ainsi qu'une couche Dropout
model_final_vgg16.add(layers.Flatten())
model_final_vgg16.add(layers.Dense(512, activation='relu'))
model_final_vgg16.add(layers.Dropout(0.2))
model_final_vgg16.add(layers.Dense(6, activation='softmax'))

# Résumé du modèle final
model_final_vgg16.summary()


#*-----------------------------------------------------*#
#*--------------- COMPILATION DU MODELE ---------------*#
#*-----------------------------------------------------*#

model_final_vgg16.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.Adam(learning_rate=0.0001),
                          metrics=['accuracy'])


#*--------------------------------------------------------*#
#*---------------- ENTRAINEMENT DU MODELE ----------------*#
#*--------------------------------------------------------*#

# Nombre d'époques pour l'entraînement
number_of_epochs = 20

# Chemin pour sauvegarder les modèles
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
vgg16_filepath = os.path.join(models_dir, 'model_vgg_16_' + '-saved-model-{epoch:02d}-acc-{val_accuracy:.2f}.hdf5')

# Rappel pour sauvegarder le meilleur modèle basé sur la précision de validation
vgg_checkpoint = callbacks.ModelCheckpoint(vgg16_filepath,
                                           monitor='val_accuracy',
                                           mode='max',
                                           save_best_only=True)

# Rappel pour arrêter l'entraînement prématurément si la perte ne diminue pas après un certain nombre d'époques
vgg_early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                             mode='min',
                                             patience=10)

# Entraînement du modèle avec les paramètres spécifiés
vgg16_history = model_final_vgg16.fit(train_generator, 
                                      epochs=number_of_epochs,
                                      validation_data=valid_generator,
                                      callbacks=[vgg_checkpoint, vgg_early_stopping],
                                      verbose=1)
# Sauvegarde du modèle
models_dir = 'models'
model_final_vgg16.save(os.path.join(models_dir, 'cnn_vgg16.h5'))

# Tracé des courbes des courbes de perte et d'exactitude
# Perte
figure = plt.gcf()
figure.set_size_inches((20, 10))
plt.title('Echantillons')
plt.xlabel('Epoque')
plt.ylabel('Entropie croisée')
plt.plot(range(1, len(vgg16_history.history['loss']) + 1), vgg16_history.history['loss'])
plt.plot(range(1, len(vgg16_history.history['val_loss']) + 1), vgg16_history.history['val_loss'])
plt.legend(['Entropie croisée train', 'Entropie croisée validation'])
plt.show()

# Exactitude 
figure = plt.gcf()
figure.set_size_inches((20, 10))
plt.title('Echantillons')
plt.xlabel('Epoque')
plt.ylabel('Exactitude')
plt.plot(range(1, len(vgg16_history.history['accuracy']) + 1), vgg16_history.history['accuracy'])
plt.plot(range(1, len(vgg16_history.history['val_accuracy']) + 1), vgg16_history.history['val_accuracy'])
plt.legend(['Précision apprentissage', 'Précision validation'])
plt.show()


#*--------------------------------------------------------------------*#
#*------------------ PREVISION SUR LES DONNEES TEST ------------------*#
#*--------------------------------------------------------------------*#

# Prétraitement des données de test
test_datagen = ImageDataGenerator(
        rescale = 1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='path',
    y_col='class_',
    batch_size=1,  # Taille de lot 1 pour les prédictions individuelles
    target_size=(224, 224),
    class_mode='categorical',  
    shuffle=False  # Ne pas mélanger les données pour que les prédictions correspondent aux étiquettes
)

#!Note! Possible de charger le modèle sauvegardé a posteriori pour effectuer les prévisions
# Charger le modèle sauvegardé
# model_final_vgg16 = models.load_model('cnn_vgg16.h5')

# Evaluation du modèle sur les données de test
test_loss, test_accuracy = model_final_vgg16.evaluate(test_generator, verbose=1)
print(f'Test loss : {test_loss:4.4f}') # Test loss : 0.6086
print(f'Test accuracy : {test_accuracy:4.4f}') # Test accuracy : 0.8368

# Faire des prédictions sur les données de test
predictions = model_final_vgg16.predict(test_generator)
# Convertir les prédictions en classes prédites (indices de classe)
predicted_classes = np.argmax(predictions, axis=1)

# Créer une matrice de confusion
true_classes = test_generator.classes  # Classes réelles des données de test
class_labels = list(test_generator.class_indices.keys())  # Étiquettes de classe
confusion_mtx = confusion_matrix(true_classes, predicted_classes)

# Affichage de la matrice de confusion sous forme de heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Classe Prédite')
plt.ylabel('Classe Réelle')
plt.title('Matrice de Confusion')
plt.show()
