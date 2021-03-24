import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
import os
import glob

def create_df():

    df = pd.read_csv('challenge/challenge_anno.txt', delimiter = "\t")

    # target variable
    y = np.array(df["Label"]).reshape(-1, 1)


    def binning_target(i):
    
        """function used to create 4 classes based on the quartlies"""
    
        if i <= 8:
            return "1"
        elif i > 8 and i <= 49:
            return "2"
        elif i > 49 and i <= 80:
            return "3"
        else:
            return "4"

    df["Categorical_Label"] = df["Label"].apply(binning_target)

    return df


# An image preprocessor that resizes the image, ignoring the aspect ratio.
class Preprocessor:
    def __init__(self, width, height):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height))

class DatasetLoader:
    def __init__(self,preprocessors = None):
         # store the image preprocessor
        self.preprocessors = preprocessors
        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []


    def load_data(self, df, verbose = -1):

        imagePaths = list((pd.Series(glob.glob('challenge/*.png'))))

        # load the images, re-size them and create labels 
        data = []
        labels = []
        cat_labels=[]
        file_names = []

        for path in imagePaths:
            image = cv2.imread(path)
            label = df.loc[path.split(os.path.sep)[-1]==df["File"]]["Label"] # get the label for each picture
            cat_label = df.loc[path.split(os.path.sep)[-1]==df["File"]]["Categorical_Label"].item() # I will use it to perform an stratified split
            file_name = df.loc[path.split(os.path.sep)[-1]==df["File"]]["File"].item() # I will use it to track the files

            if self.preprocessors is not None:
               # loop over the preprocessors and apply each to
               # the image
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            cat_labels.append(cat_label)
            file_names.append(file_name)
    
        X = np.array(data)
        y = np.array(labels)
        y_cat = np.array(cat_labels)

        return X, y, y_cat

