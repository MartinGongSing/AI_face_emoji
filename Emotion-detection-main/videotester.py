import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("best_model.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)



angry    = cv2.imread("../emoji/angry.png")
disgust  = cv2.imread("../emoji/disgust.png")
fear     = cv2.imread("../emoji/fear.png")
happy    = cv2.imread("../emoji/happy.png")
sad      = cv2.imread("../emoji/sad.png")
surprise = cv2.imread("../emoji/surprised.png")
neutral  = cv2.imread("../emoji/neutral.png")




while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:

        

        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(predicted_emotion)

        #################################
        ####### put emoji on face #######
        #################################
        emoji = neutral
        if predicted_emotion == 'angry':
            emoji = angry
        elif predicted_emotion == 'disgust':
            emoji = disgust
        elif predicted_emotion == 'fear':
            emoji = fear
        elif predicted_emotion == 'happy':
            emoji = happy
        elif predicted_emotion == 'neutral':
            emoji = neutral
        elif predicted_emotion == 'sad':
            emoji = sad
        elif predicted_emotion == 'surprise':
            emoji = surprise
        else :
            emoji = neutral


        w1 = w/2
        h1 = h/2
        # size emoji 
        dim = (w,h)
        resized = cv2.resize(emoji, dim)
        

        if test_img[x:x+w, y:y+h,:].shape !=  resized[0:w,0:h,:].shape: 
            added_img = test_img
            # print("Face not detected")
        else : 
            # print("Face detected")
            added_img = cv2.addWeighted(test_img[x:x+w, y:y+h,:], 0, resized[0:w,0:h,:], 1, 0)


        
            
            test_img[y:y+h, x:x+w,:] = added_img         # Comment this to remove the emoji

        #################################
        #################################
        #################################




    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows