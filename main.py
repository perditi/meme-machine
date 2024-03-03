#authored by Meriem Mostefai for uOttahack 6
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

#loading the existing trained model
model = load_model('trained_meme_model.h5')

#loading the image
img = image.load_img("kitty.jpeg",target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

#now for inference!
predictions = model.predict(img_array)

#interpretation
predicted_class = np.argmax(predictions[0])
print("Predicted class:", predicted_class)