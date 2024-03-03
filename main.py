#Made with the help of ______________________________
#Edited by Jordan Lau

from PIL import Image
filename = "cat.jpg"
img = Image.open(filename)
img.load()
img = img.resize((32, 32))

from numpy import array
import matplotlib.pyplot as plt
from keras.datasets import cifar10 #dataset with color images
from keras.utils import to_categorical

(training_images, training_labels), _ = cifar10.load_data()

#convert image data, while overriding the test data
training_images = training_images.astype("float32") / 255
test_images = (array([img])).astype("float32") / 255 #test_images.astype("float32") / 255

training_labels = to_categorical(training_labels, 10)
test_labels = to_categorical(array([3]), 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# DEFINE THE ML MODEL

#initializes a layered model
model = Sequential()

# these images are 32x32 px

# add functions Conv2D, MaxPooling2D, Dropout, Flatten and Dense to the dataset
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()

#TRAIN THE MODEL
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(training_images, training_labels, batch_size=64, epochs=50, validation_data=(test_images, test_labels))

#PLOT THE DATA
#accuracy = % correct
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

#loss = discrepancy between training labels and test labels
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])