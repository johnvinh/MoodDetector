# program inspired by http://sefiks.com/2018/01/01/facial-expression-recognition-with-keras/
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt


# read data from kaggle dataset
def read_data():
    global x_train, y_train, x_test, y_test  # define train and test set

    x_train, y_train, x_test, y_test = [], [], [], []

    with open(r"C:\Users\peter\Desktop\EmotionRecognitionApp\emotionData\fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)  # convert content into numpy array
    num_instances = lines.size  # define the number of instances
    num_classes = 7  # one class for each emotion

    # transfer train and test set data
    for i in range(1, num_instances):  # go through every line of data
        try:
            emotion, img, usage = lines[i].split(",")  # split data into subsets
            val = img.split(" ")  # define the value of each pixel
            pixels = np.array(val, 'float32')  # define a numpy array from each pixel value
            emotion = keras.utils.to_categorical(emotion, num_classes)  # categorize different emotions from test data

            print('working')
            # depending on type of usage data (training or testing) append to array
            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif 'PublicTest' in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("", end="")

    # transform train and test sets into numpy arrays
    x_train = np.array(x_train, 'float32')
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32')
    y_test = np.array(y_test, 'float32')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype('float32')


def train_neural_network():
    model = Sequential()  # create model

    # add layers to CNN

    # PARAMETERS: num nodes, kernel size (size of filter matrix), activation function, input shape (according
    # to kaggle, input shape is 48x48 gray scale images)

    #LAYER 1:
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    #LAYER 2:
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # uses average pooling to keep small, less noticeable facial features
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    #LAYER 3:
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1)))

    #Connect Layers
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

    # display model summary
    model.summary()

    # save model as an h5 file (so that we can load it in versions)
    model.save(r'C:\Users\peter\Desktop\EmotionRecognitionApp\recognition2.h5')

    # test model with an image (Gordon Ramsay - angry)
    img = image.load_img(r"C:\Users\peter\Desktop\EmotionRecognitionApp\gordon.jpg", color_mode='grayscale',
                         target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);

    plt.gray()
    plt.imshow(x)
    plt.show()


# function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()


read_data()
train_neural_network()
