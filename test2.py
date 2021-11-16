import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

emoji = cv2.imread("drop.jpg") #emoji



while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    ####### Draw on each face #######

    for (x, y, w, h) in faces:

        w1 = w/2
        h1 = h/2
        # size emoji 

        dim = (w,h)
        resized = cv2.resize(emoji, dim)


        # show face in square
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        ##### find eyes #####

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
       
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        ####### put emoji on face #######

        print(img[x:x+w, y:y+h,:].shape)
        print(resized[0:w,0:h,:].shape)

        if img[x:x+w, y:y+h,:].shape !=  resized[0:w,0:h,:].shape: 
            added_img = img
            print("Face not detected")
        else : 
            print(faces)
            print("Face detected")
            added_img = cv2.addWeighted(img[x:x+w, y:y+h,:], 0, resized[0:w,0:h,:], 1, 0)


        
            
            img[y:y+h, x:x+w,:] = added_img


        

    # Display

    cv2.imshow('img', img)

    # cv2.imshow('new_img', new_img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()