#authored by Meriem Mostefai and Jordan Lau for uOttahack 6

import os

import threading
import time

import tkinter as tk
from tkinter.filedialog import askopenfile

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from images_file import RESOLUTION

year = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

def predict_using_ml(filename):
    #loading the existing trained model
    model = load_model('100px_meme_model.h5')

    #loading the image
    img = image.load_img(filename,target_size=(RESOLUTION, RESOLUTION))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    #now for inference!
    predictions = model.predict(img_array)

    #interpretation
    predicted_class = np.argmax(predictions[0])
    month = year[predicted_class]
    result = tk.Label(text="Predicted month: " + month + " of 2012")
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
title.configure(fg="green", font=("Courier", 60, "bold"))
title.pack()

greeting = tk.Label(text="Select a meme file!\nI'll tell ya when its from.")
greeting.configure(font=("Arial", 24))
greeting.pack()

def on_button_click():
    file = askopenfile(mode='r',filetypes=[
        ('Image Files','*png'),
        ('Image Files','*jpeg'),
        ('Image Files','*jpg')
    ])

    filename = file.name

    #try a prediction with the ML model
    threading.Thread(target=wait_messages).start()
    threading.Thread(target=predict_using_ml, args = (filename,)).start()

button = tk.Button(window, text="Try it out", command=on_button_click)
button.configure(font=("Arial",24))
button.pack()

window.mainloop()