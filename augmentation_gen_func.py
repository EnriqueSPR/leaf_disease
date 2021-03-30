# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np



def aug_generator(number, inpath, outpath, track):
    # load the input image, convert it to a NumPy array, and then
    # reshape it to have an extra dimension

    image = load_img(inpath)
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

    imageGen = datagen.flow(image, batch_size=1, save_to_dir=outpath,
                        save_prefix=track, save_format="png")

    # loop over examples from our image data datagenmentation generator
    for image in imageGen:
        # increment our counter
        total += 1

        # if we have reached n examples, break from the loop
        if total == number:
            break