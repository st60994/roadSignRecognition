import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from skimage.feature import hog

def extract_hog_features(image_path, cell_size=(8, 8), block_size=(2, 2), nbins=9):

    # Extrakce HOG vlastností
    hog_features = hog(small_image, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size, block_norm='L2-Hys')

    return hog_features

def sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def laplace(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F)

def prewitt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    prewitt_x = cv2.filter2D(gray, -1, kernel_x)
    prewitt_y = cv2.filter2D(gray, -1, kernel_y)
    return cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

def scharr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)

def apply_custom_filter(image, kernel):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.filter2D(gray, -1, kernel)
    return filtered_image


size = 64

# Načtení dat
train_dir = 'Dataset\\training'
val_dir = 'Dataset\\validation'
test_dir = 'Dataset\\testing'

train_data = []
train_labels = []
val_data = []
val_labels = []

# Načtení trénovacích dat
for subdir in os.listdir(train_dir):
    if os.path.isdir(os.path.join(train_dir, subdir)):
        for file in os.listdir(os.path.join(train_dir, subdir)):
            img = cv2.imread(os.path.join(train_dir, subdir, file))
            img = cv2.resize(img, [size, size])
            img = cv2.cvtColor()
            features = extract_hog_features(img)
            train_data.append(features)
            train_labels.append(subdir)

# Načtení validačních dat
for subdir in os.listdir(val_dir):
    if os.path.isdir(os.path.join(val_dir, subdir)):
        for file in os.listdir(os.path.join(val_dir, subdir)):
            img = cv2.imread(os.path.join(val_dir, subdir, file))
            img = cv2.resize(img, [size, size])
            features = extract_hog_features(img)
            val_data.append(features)
            val_labels.append(subdir)

# Příprava trénovacích dat a labelů pro klasifikaci
train_data = np.array(train_data).astype(np.float32)
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(size, size), batch_size=1, class_mode='categorical')

# Příprava validačních dat a labelů pro klasifikaci
val_data = np.array(val_data).astype(np.float32)
val_labels = np.array(val_labels)
val_labels = to_categorical(val_labels)

# Normalizace dat na rozsah [0, 1]
train_data = (train_data - np.min(train_data)) / (np.max(train_data) - np.min(train_data))
val_data = (val_data - np.min(val_data)) / (np.max(val_data) - np.min(val_data))

features_size = np.shape(features)
print(features_size)

# Vytvoření FFNN modelu
model = Sequential()
model.add(Flatten(input_shape=(size, size, 1)))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(train_labels.shape[1], activation='softmax'))

# Kompilace modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trénování modelu na edge feature
history = model.fit(train_data, train_labels, batch_size=16, epochs=100, validation_data=(val_data, val_labels))

test_loss, test_acc = model.evaluate(test_generator)
print('Testovací přesnost:', test_acc)

# Uložení modelu
model.save('model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ztrátová funkce')
plt.ylabel('Ztrátová funkce')
plt.xlabel('Epochy')
plt.legend(['trénování', 'validace'], loc='upper right')
plt.show()
