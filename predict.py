# import libraries
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from cv2 import imread, resize
from pickle import load
import numpy as np
import matplotlib.pyplot as plt




def predictor(img_file_path, model_path, scaler_path):
    #load model and pickle
    with open(scaler_path, 'rb') as pickle_scaler:
        scaler = load(pickle_scaler)
    model = load_model(model_path)
    im =imread(img_file_path)
    img = resize(im, (299, 299))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    processed_pic =  preprocess_input(img_array_expanded_dims)

    plt.rcParams["figure.figsize"] = (5,5)
    plt.figure()
    plt.imshow(im)
    plt.colorbar()
    plt.grid(False)

    plt.show()

    pred = float(scaler.inverse_transform(model.predict(processed_pic))[0][0])

    return "The predicted disease score is: {}".format(round(pred,2))