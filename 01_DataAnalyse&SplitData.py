#*---------------------------------------------------------------*#
#*------------------- ANALYSE & SPLIT DATASET -------------------*#
#*---------------------------------------------------------------*#

#!Préambule! Chargement des librairies & chemin de répertoire
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

# Chemin du répertoire contenant les sous-répertoires (classes)
DATA_PATH = './data_input/garbage_classification_6classes/garbage_6/'

#*------------------- PRE-ANALYSE DATASET -------------------*#

#*-------- Volumétrie des classes  --------*#
# Liste des classes (noms des sous-répertoires)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# ou
classes = os.listdir(DATA_PATH) 

# Création de l'objet dic() pour stocker le nombre d'images par classe
image_counts = {}

### BOUCLE FOR ### 
# Parcours des classes
for class_name in classes:
    # Chemin du sous-répertoire spécifique à la classe
    class_path = os.path.join(DATA_PATH, class_name)
    
    # Liste des fichiers dans le sous-répertoire
    files = os.listdir(class_path)

    # Nombre d'images dans la classe
    image_count = len(files)
    
    # Stockage du nombre d'images dans le dictionnaire
    image_counts[class_name] = image_count
    
    # Stockage du nombre d'images dans le dictionnaire
    image_counts[class_name] = image_count

# Affichage du nombre d'images par classe
for class_name, count in image_counts.items():
    print(f"Classe {class_name}: {count} images")
# Résultats :
# Classe cardboard: 403 images
# Classe glass: 501 images
# Classe metal: 410 images
# Classe paper: 594 images
# Classe plastic: 482 images
# Classe trash: 137 images
# --> 2527 images

#*-------- Visualisation des classes  --------*#
# Affichage de 5 images aléatoires pour chaque classe
for class_name in classes:
    
    # Sélection aléatoire de 5 fichiers
    random_files = random.sample(files, 5)
    
    # Affichage des images
    plt.figure(figsize=(15, 5))
    for i, file_name in enumerate(random_files, 1):
        file_path = os.path.join(class_path, file_name)
        img = mpimg.imread(file_path)
        plt.subplot(1, 5, i)
        plt.imshow(img)
        plt.title(f'{class_name}')
        plt.axis('off')
    plt.show()


#*------------------- PREPARATION DES DONNEES -------------------*#

#*-------- Conversion en dataframe --------*#
# Création de l'objet list() pour stocker les chemins des images et leurs classes
data = []

### BOUCLE FOR ###
# Parcours de chaque classe
for class_name in classes:
    # Chemin du sous-répertoire spécifique à la classe : Ex ici ./data_input/garbage_classification_6classes/garbage_6/trash'
    class_path = os.path.join(DATA_PATH, class_name)
    
    # Liste des fichiers dans le sous-répertoire
    files = os.listdir(class_path)
    
    # Ajout des chemins des images et de leurs classes dans la liste
    for file_name in files:
        file_path = os.path.join(class_path, file_name)
        data.append((file_path, class_name))

# Création du DataFrame
df = pd.DataFrame(data, columns=['path', 'class_'])

#*-------- Création des ensembles train / test / validation --------*#
#!Remarque! Nécessaire de mélanger le dataframe avant de spliter 
df = df.sample(frac=1) #~ arg 'frac=1' -> toutes les lignes sont mélangées de manière aléatoire

# Split du DataFrame en ensemble d'entraînement (70%), de validation (15%) et de test (15%)
train_df, temp_df = train_test_split(df, test_size=0.3)
valid_df, test_df = train_test_split(temp_df, test_size=0.5)


print(train_df["class_"].value_counts())
print('total d\'images de l\'ensemble d\'entraînement:', len(train_df)) 
#!Résultat! total d'images de l'ensemble d'entraînement : 1768

print(valid_df["class_"].value_counts())
print('total d\'images de l\'ensemble de validation:', len(valid_df)) 
#!Résultat! total d'images de l'ensemble de validation: 379

print(test_df["class_"].value_counts())
print('total d\'images de l\'ensemble de test:', len(test_df)) 
#!Résultat! total d'images de l'ensemble de test: 380


#*-------- Sauvegarde des datasets --------*#
# Chemin où sauvegarder les fichiers CSV
save_path = 'data_input/garbage_classification_6classes'

# Sauvegarde de l'ensemble d'entraînement
train_save_path = os.path.join(save_path, 'train_df.csv')
train_df.to_csv(train_save_path, index=False)

# Sauvegarde de l'ensemble de validation
valid_save_path = os.path.join(save_path, 'valid_df.csv')
valid_df.to_csv(valid_save_path, index=False)

# Sauvegarde de l'ensemble de test
test_save_path = os.path.join(save_path, 'test_df.csv')
test_df.to_csv(test_save_path, index=False)