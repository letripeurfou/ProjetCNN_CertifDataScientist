import os
import sys
from keras import models


print(sys.argv[1:])
uploaded_picture = ""


def predict(uploaded_picture):
    #os.path.dirname
    model_path = os.path.join("models", "cnn_vgg16_v1.h5")
    #model_vgg16 = models.load_model(model_path)
    result = "carton"
    return result

print(predict(uploaded_picture))