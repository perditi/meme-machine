from PIL import Image
import numpy as np
import os

# Path to the directory containing the images
images_folder = "C:\\Users\\Owner\\Dropbox\\PC\\Desktop\\file\\meme-machine\\images"

# Path to the newly generated folder for the text files
output_folder = "C:\\Users\\Owner\\Dropbox\\PC\\Desktop\\file\\meme-machine\\output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Traverse the images folder
for filename in os.listdir(images_folder):
    # Check if the file is an image (you can add more extensions if needed)

    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Load the image
        img = Image.open(os.path.join(images_folder, filename))
        
        # Convert the image to a NumPy array
        img_array = np.array(img)

       # Reshape the 3D array into a 2D array
img_array_2d = img_array.reshape(-1, img_array.shape[-1])

# Save the reshaped array as a text file
np.savetxt(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt"), img_array_2d, fmt='%d')
   

print("Images converted to NumPy arrays and saved as text files.")
