from PIL import Image
import numpy as np
import os

# Jan is 0, Feb is 1, ... , Dec is 11

# a constant variable so i don't have to keep changing stuff to experiment
RESOLUTION = 100

def get_trainable_sets():
    # Path to the directory containing the images
    images_folder = "images"

    # The output array
    output_imgset = []
    output_labelset = []

    # Traverse the images folder
    for filename in os.listdir(images_folder):
        #print(filename)
        # Check if the file is an image (you can add more extensions if needed)
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Load the image
            img = Image.open(os.path.join(images_folder, filename))

            # Resize the image to 150x150 
            img = img.resize((RESOLUTION,RESOLUTION))
            
            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Append the current image's nparray to the overall output array
            output_imgset.append(img_array)
        
        # Labels the image with the month it is from
        if filename.startswith("jan"):
            label = 0
        elif filename.startswith("feb"):
            label = 1
        elif filename.startswith("mar"):
            label = 2
        elif filename.startswith("apr"):
            label = 3
        elif filename.startswith("may"):
            label = 4
        elif filename.startswith("jun"):
            label = 5
        elif filename.startswith("jul"):
            label = 6
        elif filename.startswith("aug"):
            label = 7
        elif filename.startswith("sep"):
            label = 8
        elif filename.startswith("oct"):
            label = 9
        elif filename.startswith("nov"):
            label = 10
        elif filename.startswith("dec"):
            label = 11
            
        output_labelset.append(label)
        
    temp1 = np.array(output_imgset)
    temp2 = np.array(output_labelset)
    print("Successful data parsing")
    return temp1, temp2
