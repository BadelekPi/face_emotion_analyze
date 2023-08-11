# importowanie bibliotek
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import pandas as pd
import glob

# # odczyt bazy danych FER2013
# csv_file = pd.read_csv("fer2013.csv")
# # wstępne przetwarzanie danych oraz usunięcie podzbioru emocji wstrętu
# csv_file = csv_file.loc[csv_file['emotion'] != 1]
# faces = csv_file['pixels'].tolist()
# emotions = csv_file['emotion'].tolist()

# # przesunięcie etykiet
# for i in range(len(emotions)):
#   if emotions[i] != 0:
#     emotions[i] -= 1

# # konwersja na typ float32
# for i in range(len(faces)):
#   faces[i] = np.array(faces[i].split()).reshape(48, 48).astype('float32')

# faces = np.array(faces)
# emotions = np.array(emotions)
# X = list(faces)
# y = list(emotions)

# def get_folder_name(subdir):
#     dir = subdir.split("/")[-1]
#     return dir

# def get_emotion_id(subdir):
#     emotion = get_folder_name(subdir)
#     try:
#         return emotions.index(emotion)+1 # poniewaz chce index zaczynajac od 1
#     except:
#         raise Exception("No match folder name to any emotion") 
    
# # odczyt własnej bazy danych i wstępne przetwarzanie danych
# rootdir = "dataset"
# emotions = ["Neutralny", "Szczescie", "Smutek", "Zlosc", "Zaskoczenie", "Wstret", "Strach"]


# # for subdir, dirs, images in os.walk(rootdir):
# #    for image in images:
# #       # sprawdzenie czy obraz jest w formacie JPG
# #         if (image.endswith(".jpg")):
# #             # regex dla sprawdzenia pierwszych dwóch znaków
# #             # w przeciwnym razie nadpisanie nazwy pliku: {id_emocji}_{nazwa_pliku}
# #             regex = r'^[1-7]_'
# #             if not re.match(regex, image):
# #               os.rename(f"{subdir}/{image}", f"{subdir}/{get_emotion_id(subdir)}_{image}")
# #         list_files.append(f"{subdir}/{image}")

# # stworzenie listy ze wszystkimi ścieżkami próbek danych
# images = np.array([os.path.join(folder, filename)
#             for folder, _, filenames in os.walk(rootdir)
#             for filename in filenames])
# # przetasowanie danych
# np.random.shuffle(images)

# for image in images:
#     try:
#       img = cv2.imread(image)
#       img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#       dim = (48, 48)
#       img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#       parent_name = os.path.basename(os.path.dirname(image))
#       X.append(img)
#       y.append(emotions.index(parent_name)-1) #poniewaz chce index od 0
#     except:
#         print(image)

# dataset = np.array(X)
# y = np.array(y)

# # podział zbioru danych na dane uczący oraz testujące
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.10)
# X_train1, X_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test
# X_train = np.array(X_train)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# # konwersja typu danych na float32
# def convert_dtype(x):
#     """
#     Inputs:
#         x: numpy array
#     Outputs:
#         x_float: numpy array, dtype of elements changed to 'float32'
#     """
#     x_float = x.astype('float32')
#     return x_float

# X_train = convert_dtype(X_train)
# X_test = convert_dtype(X_test)

# # normalizacja
# def normalize(x):
#     """
#     Inputs:
#         x: numpy array
#     Outputs:
#         x_n: numpy array, elements normalized to be between (0, 1)
#     """
#     x_n = (x - 0)/(255)
#     return x_n

# X_train = normalize(X_train)
# X_test = normalize(X_test)

# def save_3d_array(array):
#     stacked = pd.Panel(array.swapaxes(1,2)).to_frame().stack().reset_index()
#     stacked.columns = ['x', 'y', 'z', 'value']
#     # zapis na dysku
#     stacked.to_csv(f'stacked_{array}.csv', index=False)

# # zmiana wymiaru danych
# def reshape(x):
#     """
#     We need to reshape our train_data to be of shape (samples, height, width, channels) pass to Conv2D layer of keras
#     Inputs:
#         x: numpy array of shape(samples, height, width)
#     Outputs:
#         x_r: numpy array of shape(samples, height, width, 1)
#     """
#     # YOUR CODE HERE
#     x_r = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
#     return x_r

# X_train = reshape(X_train)
# X_test = reshape(X_test)

# # dla sprawdzenia zapisu danych
# np.save("X_train", X_train)
# np.save("X_test", X_test)


# # Konwersja na wektory ,,one-hot''
# def oneHot(y, Ny):
#     """
#     Inputs:
#         y: numpy array if shape (samples, ) with class labels
#         Ny: number of classes
#     Outputs:
#         y_oh: numpy array of shape (samples, Ny) of one hot vectors
#     """
#     # YOUR CODE HERE
#     y_oh = np.zeros((len(y), Ny))
#     for i in range(len(y)):
#       y_oh[i][y[i]] = 1
#     return y_oh

# y_train = oneHot(y_train, 6)
# y_test = oneHot(y_test, 6)
# np.savetxt("y_train", y_train, delimiter=",")
# np.savetxt("y_test", y_test, delimiter=",")

X_test = np.load("X_test.npy")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Zaimportowanie wymaganych bibliotek do stworzenia modelu sieci
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential # reprezentacja warst sieci
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

def create_model():
    """
    Inputs:
        None
    Outputs:
        model: compiled keras model
    """
    def model_emotion(input_shape, num_classes, l2_regularization=0.01):
        regularization = l2(l2_regularization)

        # base
        img_input = Input(input_shape)
        """
        Dodawanie warstw konwolucyjnych do modelu przy wykorzystaniu funkcji Conv2D, gdzie:
        filters: liczba filtrow/kanalow ktore chcemy uzyc w warstwie
        kernel_size: rozmiar filtrow 3x3
        activation: funkcja aktywacji
        input_shape: ksztalt wejsciowy obrazow (szerokosc, wysokosc i liczba kanalow)
        strides: krok przesuwu okna konwolucyjnego
        kernel_regularizer: wprowadzenie regularyzacji L1 lub L2 do wag jadra (kernel)
        use_bias: okresla czy warstwa konwolucyjna ma uzywac obciazenia wraz z wagami jadra
        """
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(img_input)
        # BatchNormalization sluzy do normalizacji warstw w sieciach neuronowych
        x = BatchNormalization()(x)
        # Stosowanie activation relu po warstwie konwolucyjnej, aby wprowadzic nieliniowosc
        x = Activation('relu')(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
                   use_bias=False)(x)
        # normalizacja wsadowa
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # module 1
        # padding='same' dodaje piksele tak aby wielkosc wyjsciowej macierzy byla taka sama jak macierz wejsciowa
        residual = Conv2D(16, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)
        # separableconv2D - najpierw stosuje konwolucje przestrzenna a pozniej konwolucje punktowa
        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(16, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        # podzielenie obrazu na prostokatne podregiony i wybranie maksymalnej wartosci
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 2
        residual = Conv2D(32, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(32, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 3
        residual = Conv2D(64, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(64, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        # module 4
        residual = Conv2D(128, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3, 3), padding='same',
                            kernel_regularizer=regularization,
                            use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = layers.add([x, residual])

        x = Conv2D(num_classes, (3, 3),
                   # kernel_regularizer=regularization,
                   padding='same')(x)
        # zastosowanie sredniej wartosci na calym obszarze macierzy wejsciowej
        x = GlobalAveragePooling2D()(x)
        # wyjscie: prawdopodobienstwo przynaleznosci danego wejscia do kazdej z klas
        output = Activation('softmax', name='predictions')(x)

        model = Model(img_input, output)
        return model
    model = model_emotion((48, 48, 1), 6)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model
# utworzenie instancji modelu
model = create_model()

# training the model
history = model.fit(X_train, y_train, validation_split = 0.1, epochs=20, batch_size=200)

#creating functions to predict the output using our model
def predict(x):
    """
    Inputs:
        x: input samples
        model: keras model
    Outputs:
        y: predicted labels
    """
    # YOUR CODE HERE
    y = model.predict(x)
    y1 = np.zeros(y.shape)
    for i in range(len(y)):
      y1[i][np.argmax(y[i])] = 1
    y = y1
    return y

# function to create one-hot vectors into labels
def oneHot_tolabel(y):
    """
    Inputs:
        y: numpy array of shape (samples, Ny)
    Outputs:
        y_b: numpy array of shape (samples,) where one hot encoding is converted back to class labels
    """
    # YOUR CODE HERE
    y_b = []
    for i in range(len(y)):
      y_b.append(np.argmax(y[i]))

    y_b = np.array(y_b)
    return y_b

#function to create confusion matrix

def create_confusion_matrix(true_labels, predicted_labels):
    """
    Inputs:
        true_labels: numpy array of shape (samples, ) with true_labels
        test_labels: numpy array of shape(samples, ) with test_labels
    Outputs:
        cm: numpy array of shape (Ny, Ny), confusion matrix. Ny -> number of unique classes in y
    """
    # YOUR CODE HERE
    cm = np.zeros((7, 7))
    for i in range(len(true_labels)):
      if true_labels[i] == predicted_labels[i]:
        cm[true_labels[i]][true_labels[i]] = cm[true_labels[i]][true_labels[i]] + 1
      elif true_labels[i] != predicted_labels[i]:
        cm[true_labels[i]][predicted_labels[i]] += 1
    return cm

#predicting the output using model
predicted_labels_train = oneHot_tolabel(predict(X_train))

#creating confusion matrix
cm = create_confusion_matrix(oneHot_tolabel(y_train), oneHot_tolabel(predicted_labels_train)).astype(int)

#printing confusion matrix

print(cm)

#visualising the training of the model

history.history.keys()
import matplotlib.pyplot as plt
plt.plot(range(len(history.history['val_acc'])), history.history['val_acc'])
plt.show()

# testing the accuracy of the model
def accuracy(x_test, y_test, model):
    """
    Inputs:
        x_test: test samples
        y_test : test labels
        model: keras model
    Ouputs:
        acc: float, accuracy of test data on model
    """
    # YOUR CODE HERE
    acc = model.evaluate(x_test, y_test)[1]
    return acc

acc = accuracy(X_test, y_test, model)
print('Test accuracy is, ', acc*100, '%')