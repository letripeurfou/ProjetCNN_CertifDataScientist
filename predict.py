import os
import sys
import numpy as np
from keras import models, utils
from keras.applications.vgg16 import preprocess_input

path = 'C:/Users/D58880/Documents/PyWorkSpace/ProjetCNN_CertifDataScientist/'

# Chargement image
img_path = sys.argv[1]
img = utils.load_img(img_path, target_size=(224, 224))
x = utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Chargement modele pré-entrainé
model_path = os.path.join(path + "models/cnn_vgg16_v1.h5")
model_final_vgg16 = models.load_model(model_path)

# Prévision
predictions = model_final_vgg16.predict(x)
# Récupérer l'indice de la classe avec la probabilité la plus élevée
predicted_class_index = np.argmax(predictions)

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Mappage l'indice de la classe à l'étiquette correspondante
predicted_class_label = classes[predicted_class_index]
print(predicted_class_label)
