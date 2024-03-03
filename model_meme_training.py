#Made with the help of https://reintech.io/blog/how-to-create-an-image-recognition-system-with-python
#Code by Timothy Mao, Meriem Mostefai, Zahra Suleymanova, and Jordan Lau

import images_file

training_images, training_labels = images_file.get_trainable_sets()

#convert image data, while overriding the test data
training_images = training_images.astype("float32") / 255
print(training_images.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout



# DEFINE THE ML MODEL

#initializes a layered model
model = Sequential()


# add functions/layers Conv2D, MaxPooling2D, Dropout, Flatten and Dense to the dataset
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(images_file.RESOLUTION, images_file.RESOLUTION, 3)))
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
model.add(Dense(12, activation="softmax"))

model.summary()

#TRAIN THE MODEL
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(training_images, training_labels, epochs=50)

#SAVING THE TRAINED MODEL
model.save("100px_meme_model.h5")
