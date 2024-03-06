#*---------------------------------------------------------------*#
#*------------------- CNN PRE-TRAINED - VGG16 -------------------*#
#*---------------------------------------------------------------*#

#!Preambule! Importation des librairies
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers, models, callbacks, optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#*-------------------------------------------------------------*#
#*------------------ PREPARATION DES DONNEES ------------------*#
#*-------------------------------------------------------------*#

#*-------- Importation des dataframes --------*#
data_dir = 'data_input/garbage_classification_6classes'

# Ensemble d'entraînement
train_path = os.path.join(data_dir, 'train_df.csv')
train_data = pd.read_csv(train_path)
print('total d\'images de l\'ensemble d\'entraînement:', len(train_data)) # 2 188

# Ensemble de validation
valid_path = os.path.join(data_dir, 'valid_df.csv')
valid_data = pd.read_csv(valid_path)
print('total d\'images de l\'ensemble d\'entraînement:', len(valid_data)) # 469

# Ensemble de test
test_path = os.path.join(data_dir, 'test_df.csv')
test_data = pd.read_csv(test_path)
print('total d\'images de l\'ensemble d\'entraînement:', len(test_data)) # 469

#*----------------------------------------------------------*#
#*--------------- PRE-TRAITEMENT DES DONNEES ---------------*#
#*----------------------------------------------------------*#

#*-------- ENSEMBLE D'ENTRAINEMENT --------*#
# Création d'un générateur de données pour l'ensemble d'entraînement
train_datagen = ImageDataGenerator(
    ### Pré-traitement des images ###
    preprocessing_function=preprocess_input,
    ### Augmentation des données ###
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
    )

# Création d'un générateur d'images à partir du DataFrame d'entraînement
train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    x_col='path',
    y_col='class_',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical'
    )

#*-------- ENSEMBLE DE VALIDATION --------*#
#!!!Remarque!!! Les données de l'ensemble de validation ne doivent pas être augmentées
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Création du générateur d'images 'validationGenerator'
valid_generator = valid_datagen.flow_from_dataframe(
    dataframe = valid_data,
    x_col='path',
    y_col='class_',
    batch_size=32,
    target_size=(224, 224),
    class_mode="categorical"
    )

#*------------------------------------------------------*#
#*--------------- PONDERATION DE CLASSES ---------------*#
#*------------------------------------------------------*#

#!Note! Poids de classe pour accorder plus d'importance aux classes sous-representees
# Calcul des poids : total samples / (nb of classes/samples in the class)

# Extraction des noms de classes
class_labels = train_data['class_'].unique()

# Calcul des poids de classe
weights = compute_class_weight(class_weight='balanced',
                               classes=class_labels,
                               y = train_data['class_'])

# Conversion en objet dict() en vue de l'entraînement du modèle
class_weights = dict(zip(train_generator.class_indices.values(), weights))

#*-----------------------------------------------------------*#
#*--------------- MODELE PRE-ENTRAINE - VGG16 ---------------*#
#*-----------------------------------------------------------*#

# Instanciation de la base de convolution du modèle VGG16
model_vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224,224,3))

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

### BOUCLE FOR ####
for layer in model_vgg16.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable

print('Nombre de poids entraînables pour le dernier block5:',
      len(model_vgg16.trainable_weights))

# Couches post fine-tuning
for layer in model_vgg16.layers:
    if layer.trainable:
        print("Couche '{}' entrainable".format(layer.name))
    else:
        print("Couche '{}' non entrainable".format(layer.name))


#*----------------------------------------------------*#
#*-------------- ARCHITECTURE DU RESEAU --------------*#
#*----------------------------------------------------*#

model_final_vgg16 = models.Sequential()

# Ajout du modele VGG16
model_final_vgg16.add(model_vgg16)

# Ajout d'un classifieur entièrement connecté ainsi qu'une couche Dropout
model_final_vgg16.add(layers.Flatten())
model_final_vgg16.add(layers.Dense(512, activation = 'relu'))
model_final_vgg16.add(layers.Dropout(0.2))
model_final_vgg16.add(layers.Dense(6, activation = 'softmax'))

# Résumé du modèle final
model_final_vgg16.summary()


#*-----------------------------------------------------*#
#*--------------- COMPILATION DU MODELE ---------------*#
#*-----------------------------------------------------*#

model_final_vgg16.compile(loss = 'categorical_crossentropy',
                          optimizer = optimizers.Adam(learning_rate=0.0001),
                          metrics = ['accuracy'])


#*--------------------------------------------------------*#
#*---------------- ENTRAINEMENT DU MODELE ----------------*#
#*--------------------------------------------------------*#

# Nombre d'époques pour l'entraînement
number_of_epochs = 50

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
                                             patience=15)

# Entraînement du modèle avec les paramètres spécifiés
vgg16_history = model_final_vgg16.fit(train_generator,
                                      epochs = number_of_epochs,
                                      validation_data = valid_generator,
                                      class_weight = class_weights,
                                      callbacks = [vgg_checkpoint, vgg_early_stopping],
                                      verbose=1)
# Sauvegarde du modèle
models_dir = 'models'
model_final_vgg16.save(os.path.join(models_dir, 'cnn_vgg16_v1.h5'))

# Tracé des courbes de perte et d'exactitude
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
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
#~ Charger le modèle sauvegardé
#~ model_final_vgg16 = models.load_model('cnn_vgg16.h5')

# Evaluation du modèle sur les données de test
test_loss, test_accuracy = model_final_vgg16.evaluate(test_generator, verbose=1)
print(f'Test loss : {test_loss:4.4f}') # Test loss : 0.55
print(f'Test accuracy : {test_accuracy:4.4f}') # Test accuracy : 0.90

# Prediction
predictions = model_final_vgg16.predict(test_generator)
# Conversion en classes prédites (indices de classe)
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
