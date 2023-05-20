# Import knihoven
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt


# Načtení uloženého modelu
# model = load_model('model.h5')
size = 30
classes = 43

# Vytvoření ImageDatageneratoru pro trénovací, validační a testovací data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Nastavení cesty k trénovacím, validačním a testovacím datům
train_dir = 'Dataset\\training'
val_dir = 'Dataset\\validation'
test_dir = 'Dataset\\testing'

# Načtení dat ze složek pomocí ImageDatageneratoru
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(size, size), batch_size=1, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(size, size), batch_size=1, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(size, size), batch_size=1, class_mode='categorical')


# Definice modelu
model = Sequential()
model.add(Flatten(input_shape=(size, size, 3)))
model.add(Dense(80, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(classes, activation='softmax'))

# Kompilace modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trénování modelu
history = model.fit(train_generator, epochs=20, steps_per_epoch=train_generator.n // train_generator.batch_size, validation_data=val_generator, validation_steps=val_generator.n // val_generator.batch_size)

# Uložení natrénovaného modelu
model.save('model.h5')


# Vyhodnocení modelu na testovacích datech
test_loss, test_acc = model.evaluate(test_generator)
print('Testovací přesnost:', test_acc)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Ztrátová funkce')
plt.ylabel('Ztrátová funkce')
plt.xlabel('Epochy')
plt.legend(['trénování', 'validace'], loc='upper right')
plt.show()
