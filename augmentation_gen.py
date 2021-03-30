# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory to store datagenmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="aug",
                help="output filename prefix")
ap.add_argument("-n", "--number", type=int, default=10, required=True,
                help="number of pictures generated")

args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args['image'])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct the image generator for data datagenmentation then
# initialize the total number of images generated thus far
datagen = ImageDataGenerator(
        rotation_range=180, # value from 0 to 180 rotation
        horizontal_flip=True, # flipping pictures vertically. 
        vertical_flip=True # flipping pictures horizontally.  
)
total = 0


# construct the actual Python generator
print("[INFO] generating images...")

imageGen = datagen.flow(image, batch_size=1, save_to_dir=args["output"],
                    save_prefix=args["prefix"], save_format="jpg")

# loop over examples from our image data datagenmentation generator
for image in imageGen:
    # increment our counter
    total += 1

    # if we have reached n examples, break from the loop
    if total == args["number"]:
        break