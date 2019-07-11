# Name: PET ER PHAM
# DATE: May 14 2019

# WHAT THIS VERSION ACCOMPLISHES:
# - sends cropped image to keras model to predict emotions
import cv2
import numpy as np
from keras.models import load_model
import time
import sys
import PIL
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
import keras


def find_Face():
    # cpu - gpu configuration
    config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 56})  # max: 1 gpu, 56 cpu
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # create face cascade
    faceCascade = cv2.CascadeClassifier(
        r'C:\Users\peter\PythonProjects\MoodDetectionV1\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    # use default webcam
    cap = cv2.VideoCapture(0)

    # IMPORTANT: CHANGE TO CERTAIN TIME INTERVALS TO PREVENT LAG

    # loop until user presses "Q"
        #while cv2.waitKey(1) & 0xff != ord('q'):
    # time.sleep(1)

    # capture frame using webcam
    ret, frame = cap.read()

    # remove mirror effect from camera
    frame = cv2.flip(frame, 2)

    # set webcam to grayscale (grayscale reduces unnescescary info in the image (since we are only handling
    # facial detection and emotion recognition,colours don't matter)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # store detected faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE  # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # store a cropped image of a face for use in the emotion recognition CNN
        crop_img = frame[y:y + h, x:x + w]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)  # make image gray

        # store the cropped image as jpg
        cv2.imwrite(r'C:\Users\peter\PythonProjects\MoodDetectionV4\saved_faces\image.jpg', crop_img)

    # show resulting frame
    cv2.imshow('frame', frame)
    return emotion_recognition()



    cap.release()
    cv2.destroyAllWindows()


# recognize emotion from stored image
def emotion_recognition():
    # load Keras model that was trained from version 1
    model = load_model(r'C:\Users\peter\Desktop\EmotionRecognitionApp\recognition1.h5')

    # load the stored cropped image
    img = image.load_img(r'C:\Users\peter\PythonProjects\MoodDetectionV4\saved_faces\image.jpg', color_mode='grayscale',
                         target_size=(48, 48))

    # convert image to numpy array
    x = image.img_to_array(img)

    # change dimensions to fit model
    x = np.expand_dims(x, axis=0)

    x /= 255

    # analyze emotions
    custom = model.predict(x, batch_size=len(x))
    emotions = custom[0]  # store array of emotions

    emotions = emotions.tolist()

    # reduce neutral count
    emotions[6] = emotions[6] - 0.07

    # outout emotions in console
    print("ANGER:" + str(emotions[0] * 100))
    print("DISGUST:" + str(emotions[1] * 100))
    print("FEAR:" + str(emotions[2] * 100))
    print("HAPPY:" + str(emotions[3] * 100))
    print("SAD:" + str(emotions[4] * 100))
    print("SURPRISE:" + str(emotions[5] * 100))
    print("NEUTRAL:" + str(emotions[6] * 100))

    index = emotions.index(max(emotions))  # get index of most likely emotion

    myEmotions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # the predicted emotion is the one with the largest value
    print("PREDICTED EMOTION: " + myEmotions[index])  # print most likely emotion
    print("--------------------------------------")

    return index


find_Face()
