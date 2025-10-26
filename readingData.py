import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
    
    
#
# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#
input_path = ''
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#
# Show some random training and test images 
#
images_2_show = []
titles_2_show = []
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

show_images(images_2_show, titles_2_show)

###############################################################################
import numpy as np
import matplotlib.pyplot as plt

# Asumiendo que ya tienes tus datos cargados en x_train y y_train
# x_train: numpy array de forma (n_samples, 28, 28)
# y_train: numpy array de forma (n_samples,)


x_train = np.array(x_train)
y_train = np.array(y_train)
    
# Filtrar las imágenes que contienen los dígitos 1 y 5
indices_1_5 = np.where((y_train == 1) | (y_train == 5))[0]
x_filtered = x_train[indices_1_5]
y_filtered = y_train[indices_1_5]

# Función para calcular la intensidad media
def calcular_intensidad_media(imagen):
    return np.mean(imagen)

# Función para calcular la simetría respecto a un eje vertical
def calcular_simetria(imagen):
    # Espejar la imagen respecto al eje vertical
    imagen_espejada = np.fliplr(imagen)
    # Calcular la diferencia pixel a pixel
    diferencia = np.abs(imagen - imagen_espejada)
    # La simetría puede medirse como la media de la diferencia
    return np.mean(diferencia)

# Crear listas para almacenar las características
features = []

for img in x_filtered:
    intensidad = calcular_intensidad_media(img)
    simetria = calcular_simetria(img)
    features.append([intensidad, simetria])

features = np.array(features)

# Separar las clases para graficar
clase_1 = features[y_filtered == 1]
clase_5 = features[y_filtered == 5]

# Graficar
plt.figure(figsize=(8,6))
#plt.scatter(clase_1[:,0], clase_1[:,1], label='Dígito 1', alpha=0.2)
plt.scatter(clase_5[:,0], clase_5[:,1], label='Dígito 5', alpha=0.2)
plt.scatter(clase_1[:,0], clase_1[:,1], label='Dígito 1', alpha=0.2)
plt.xlabel('Intensidad media')
plt.ylabel('Simetría')
plt.title('Características de imágenes de dígitos 1 y 5')
plt.legend()
plt.show()
