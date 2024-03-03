#authored by Meriem Mostefai and Jordan Lau for uOttahack 6

import threading
import time

import tkinter as tk

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def predict_using_ml():
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
    result = tk.Label(text="Predicted class: "+ predicted_class.astype(str))
    result.configure(font=("Arial", 24))
    result.pack()

def wait_messages():
    while(True):
        greeting.configure(text="Chatting with our AI...",fg="black")
        time.sleep(3)
        greeting.configure(text="Reminiscing about 2012...",fg="black")
        time.sleep(3)
        greeting.configure(text="Laughing at memes...",fg="black")
        time.sleep(3)
        greeting.configure(text="Upvoting on Reddit...",fg="black")
        time.sleep(3)

window = tk.Tk()
window.geometry("400x400")

title = tk.Label(text="MEME MACHINE")
title.configure(fg="blue", font=("Courier", 60))
title.pack()

greeting = tk.Label(text="Input a meme file name:")
greeting.configure(font=("Arial", 24))
greeting.pack()

inp = tk.Spinbox(window)
inp.configure(font=("Arial", 24))
inp.pack()

def on_button_click():
    filename = inp.get()

    if(filename[len(filename)-4:] != ".jpg"):
        greeting.config(text="Input a jpg file name:",fg="red")
    else:
        threading.Thread(target=wait_messages).start()
        threading.Thread(target=predict_using_ml).start()

button = tk.Button(window, text="What month was it?", command=on_button_click)
button.configure(font=("Arial",24))
button.pack()

window.mainloop()