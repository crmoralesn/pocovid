import argparse
import numpy as np
from numpy import loadtxt
from numpy import savetxt
import pandas as pd

from numpy import save
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import os
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument(
    '-s', '--fold', type=int, default='0', help='fold to take as test data'
)
ap.add_argument('-i', '--img_size', type=tuple, default=(224, 224))


args = vars(ap.parse_args())

IMG_WIDTH, IMG_HEIGHT = args['img_size']
FOLD = args['fold']



#cargamos el modelo
base_model = load_model('pocus_fold_0',custom_objects=None, compile=True, options=None)
#rescatamos la capa deseada
model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense').output)

# extrae labels desde las carpetas
dirname = os.path.join(os.getcwd(), '/Users/javierdelao/IdeaProjects/tesis/covid19_pocus_ultrasound/data/pocus_images')
imgpath = dirname + os.sep

directories=[]
for root, dirnames, filenames in os.walk(imgpath):
    for dirname in dirnames:
        directories.append(dirname)

images=[]
# AQUI ESPECIFICAMOS UNAS IMAGENES para el test
#filenames = [
#    'test_data/covid1.jpg',
#    'test_data/neu1.jpg',
#    'test_data/regular1.jpg'
#]
DATA_DIR = '../data/cross_validation/'
imagePaths = list(paths.list_images(DATA_DIR))
#csv_data = np.array([[],[]])
csv_data = []
#csv_data = np.zeros((2,3))
contador = 0

for imagePath in imagePaths:
    images=[]
    path_parts = imagePath.split(os.path.sep)
    train_test = path_parts[-3][-1]
    label = path_parts[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=3)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    images.append(image)

    X = np.array(images) / 255.0 #convierto de lista a numpy
    test_X = X.astype('float32')
    test_X = test_X / 255.

    #print(contador)
    #contador = contador + 1
    predicted_classes = model.predict(test_X)
    test_array = np.append(predicted_classes[0],imagePath)
    test_array = np.append(test_array,label)
    if train_test == str(FOLD):
        test_array = np.append(test_array,'test')
    else:
        test_array = np.append(test_array,'train')
    test_array = np.array(test_array);
    csv_data.append(test_array)
    print(len(csv_data))


#X = np.array(images) / 255.0 #convierto de lista a numpy
#test_X = X.astype('float32')
#test_X = test_X / 255.


df = pd.DataFrame(csv_data)
df.to_csv('test_data/test_model_5_sobel_filter.csv')
#predicted_classes = model.predict(test_X)
#print(predicted_classes)











#preparamos las images
#for filepath in filenames:
#    image = plt.imread(filepath,0)
#    image_resized = cv2.resize(image, (224, 224))
#    images.append(image_resized)

#X = np.array(images) / 255.0 #convierto de lista a numpy
#test_X = X.astype('float32')
#test_X = test_X / 255.

#obtenemos los resultados /64 para la capa "dense"
#predicted_classes = model.predict(test_X)
#print(predicted_classes)

#para evaluar los resultados dentro de las 3 categorias / solo valido ocupando la ultima capa del modelo
#for i, img_tagged in enumerate(predicted_classes):
#    print(filenames[i], directories[img_tagged.tolist().index(max(img_tagged))])