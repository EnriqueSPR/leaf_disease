# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt



def split_data(X, y_normal, y_cat):

    # I use stratify here = y_cat in the test split to guarantee equal distribution of the targets across the train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_normal, test_size=0.40, random_state=42, stratify=y_cat)

    # Split test set into test and val set
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

    # Inveption v3 Preprocessing
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    X_test = preprocess_input(X_test)
    
    return  X_train, X_test, X_val, y_train, y_test, y_val


    
def train(X_train, y_train, X_val, y_val, epocs=1500, learning_rate=0.00001, batch_size=16):

    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=180, # value from 0 to 180 rotation
        horizontal_flip=True, # flipping pictures vertically. 
        vertical_flip=True # flipping pictures horizontally.  
    )

    datagen.fit(X_train) 
    # callbacks
    checkpoint_cb = ModelCheckpoint("inception_v3.h5", save_best_only=True, verbose=1) # monitoring 'val_loss'
    early_stopping_cb = EarlyStopping(patience=10)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Transformation into one dimensional tensor
    x = Dense(1024, activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Regularize with dropout

    predictions = Dense(1, activation='linear')(x) # perform regression

    model = Model(inputs=base_model.inputs, outputs=predictions) #create final model
    

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=["mse", "mae"])
    print("Training...")
    batch_size = batch_size
    hist = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=X_train.shape[0] // batch_size,
                batch_size=batch_size,
                epochs=epocs, verbose=2,
                validation_data=(X_val, y_val), validation_steps=X_val.shape[0]//batch_size,
                callbacks=[early_stopping_cb, checkpoint_cb]) 
    
    # plot training
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12,6))

    plt.plot(np.arange(0, epocs), hist.history["loss"], label="train_loss")

    plt.plot(np.arange(0, epocs), hist.history["val_loss"], label="val_loss")

    plt.title("Training (Trainable params: 23,869,6019 )", fontsize=16)
    plt.xlabel("Epocs", labelpad=15, fontsize=14)
    plt.ylabel("Loss/mse", rotation="vertical", fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # plt.savefig("Training_efficientnet.png",  dpi=300, bbox_inches="tight")

    plt.legend(fontsize=14)
    plt.show()