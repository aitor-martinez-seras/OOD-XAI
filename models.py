import os
from utils import download_file_from_google_drive, unzip_file
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def create_model(model_name):
    if model_name == 'LeNet':
        model = keras.Sequential(
            [
              keras.layers.InputLayer(input_shape=(28,28,1)),
              keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",name='conv2d'),
              keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",name='conv2d_1'),
              keras.layers.MaxPooling2D(pool_size=(2, 2), name= 'max_pooling2d_1'),
              keras.layers.Dropout(0.25,name='dropout'),
              keras.layers.Flatten(name='flatten'),
              keras.layers.Dense(128,activation='relu',name='dense'),
              keras.layers.Dropout(0.25,name='dropout_1'),
              keras.layers.Dense(10, activation="softmax", name='softmax'),
            ]
        )
        # As in the github example
        def scheduler(epoch, lr):
          if epoch < 10:
            return lr
          else:
            return lr * 0.7
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    elif model_name == 'ResNet32':
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'download'])
        RESNET32_ID = '1lb1V6KmZzhLevtybnVNd0G_5OrKYoiNK'
        RESNET32_ZIP_FILE = 'resnet32.zip'
        RESNET32_DIR = 'resnet32/'
        resnet32_zip_path = os.path.join(RESNET32_DIR, RESNET32_ZIP_FILE)
        if os.path.isdir(RESNET32_DIR):
            print(f'"{RESNET32_DIR}" directory already exist, delete if to automatically download again the files')
        else:
            os.mkdir(RESNET32_DIR)
            _ = unzip_file(download_file_from_google_drive(RESNET32_ID, resnet32_zip_path), RESNET32_DIR)
        # Import the function to create the model and create it
        from resnet32.resnet_32 import create_model
        model = create_model((32, 32, 3), 10, 'ResNet32v1')
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    else:
        raise NameError('Wrong model name selected')
    print(model.summary())

    return model