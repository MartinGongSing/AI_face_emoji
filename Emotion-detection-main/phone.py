import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import time



def img_treatment(path):
    # load model
    model = load_model("best_model.h5") # --> https://www.youtube.com/watch?v=G1Uhs6NVi-M   8'45

    # find face 
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #open camera
    cap = cv2.imread(path)


    # load emoji
    angry    = cv2.imread("../emoji/angry.png")
    disgust  = cv2.imread("../emoji/disgust.png")
    fear     = cv2.imread("../emoji/fear.png")
    happy    = cv2.imread("../emoji/happy.png")
    sad      = cv2.imread("../emoji/sad.png")
    surprise = cv2.imread("../emoji/surprised.png")
    neutral  = cv2.imread("../emoji/neutral.png")
    mask     = cv2.imread("../emoji/mask.png")
    left     = cv2.imread("../emoji/left.png")
    right     = cv2.imread("../emoji/right.png")


    # previous_eye = 0 # trying to change emoji based on eye hight

    while True:

        # wait 1 sec before restarting the loop, to prevent the blink of the emoji
        # time.sleep(1)

        
        test_img = cap # captures frame and returns boolean value and captured image

        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        # puts face coord in list
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:

            #make rectangle arround face
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            #make transformations on face
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            # based on model(.h5) predicts in list the emotion catured on face
            predictions = model.predict(img_pixels)

            # find max indexed array -> most suitable
            max_index = np.argmax(predictions[0])

            #list of available emotions
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            #saves the emotion's name
            predicted_emotion = emotions[max_index]

            #puts text on top of square arround face
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            #################################
            ####### put emoji on face #######
            #################################
            
            #if nothing
            emoji = neutral
            #if emotion detected


            # if emotion2 == 'masked' : 
            #     emoji = mask
            # else : 


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
                emoji = mask



            dim = (w,h)
            resized = cv2.resize(emoji, dim)
            
            #test_img : the one captured by camera

            #if : if no faces are detected or if dim did not work -> do nothing
            #else : if faces are detected -> addWeighted : superpose emoji + camera

            if test_img[x:x+w, y:y+h,:].shape !=  resized[0:w,0:h,:].shape: 
                added_img = test_img
                # print("Face not detected")
            else : 
                # print("Face detected")
                added_img = cv2.addWeighted(test_img[x:x+w, y:y+h,:], 0, resized[0:w,0:h,:], 1, 0)

                #right axis + add of images

    ##############################################################################################
                test_img[y:y+h, x:x+w,:] = added_img         # Comment this to remove the emoji
    ##############################################################################################



        filename = "./rep/response.jpg"
        #necessary transformations for display
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imwrite(filename, resized_img)
        return "response.jpg"
        # cv2.imshow('Facial emotion analysis ', resized_img)



    #     if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
    #         break

    # # cap.release()
    # cv2.destroyAllWindows


from bottle import Bottle, request, response, template, static_file
import os


app = Bottle()

temp = """<html>
<head><title>Home</title></head>
<body>
<h1>Upload a file</h1>
<form method="POST" enctype="multipart/form-data" action="/">
<input type="file" name="uploadfile" /><br>
<input type="submit" value="Submit" />
</form>
</body>
</html>"""

image_send = """<html>
<head><title>Home</title></head>
<body>
<h1> Result </h1>
<img src={{!source}} alt="Result">
</body>
</html>"""


@app.get('/')
def index():
    return temp

@app.get('/path/<picture>')

def serve_pictures(picture):

    print(picture)

    return static_file(picture, root='./rep')

@app.post('/')
def upload():
    # A file-like object open for reading.
    upload = request.POST['uploadfile']
    name, ext = os.path.splitext(upload.filename)
    print(upload.filename)
    print(ext)
    if ext not in ('.png','.jpg','.jpeg','.JPG'):
        return 'File extension not allowed.'

    upload.save('./tpm') # appends upload.filename automatically
    pathtofile='./tpm'+ upload.filename
    print(pathtofile)
    path = img_treatment('./tpm/'+ upload.filename)
    print(path)
    return template(image_send, source ='path/response.jpg')
    




if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=8082)