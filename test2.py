import cv2

# Load the cascade ---- LBP trained
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

drop = cv2.imread("emoji\drop.jpg") #drop
wink = cv2.imread("emoji\wink.png") #wink
right = cv2.imread("emoji\right.png") #right
left = cv2.imread("emoji\left.png") #right


previous_ey = 0 ## trying to change emoji based on eye hight

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    

    ####### Draw on each face #######

    for (x, y, w, h) in faces:



        # show face in square
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        ##### find eyes #####

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
       
        eyes = eye_cascade.detectMultiScale(roi_gray)
        emoji = drop

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_c = ey + eh/2

            # trying to change emoji based on eye hight
            if (previous_ey - 0.5) <= eye_c <= (previous_ey + 0.5) : 
                emoji = wink
                print ("wink")


            else :
                emoji = drop
                print("drop")
            
            print(previous_ey, "\n", eye_c)


        previous_ey = eye_c 
            
            
        #################################
        ####### put emoji on face #######
        #################################
        
        w1 = w/2
        h1 = h/2
        # size emoji 
        dim = (w,h)
        resized = cv2.resize(emoji, dim)

        # print(img[x:x+w, y:y+h,:].shape)
        # print(resized[0:w,0:h,:].shape)

        if img[x:x+w, y:y+h,:].shape !=  resized[0:w,0:h,:].shape: 
            added_img = img
            # print("Face not detected")
        else : 
            # print("Face detected")
            added_img = cv2.addWeighted(img[x:x+w, y:y+h,:], 0, resized[0:w,0:h,:], 1, 0)


        
            
            img[y:y+h, x:x+w,:] = added_img         # Comment this to remove the emoji

        #################################
        #################################
        #################################
        

    # Display

    cv2.imshow('img', img)

    # cv2.imshow('new_img', new_img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()