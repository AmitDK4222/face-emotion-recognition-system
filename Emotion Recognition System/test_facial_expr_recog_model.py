from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#Load Haar-Cascade Frontal Face detector in face_classifier variable
face_classifier = cv2.CascadeClassifier(r'C:\Users\amit1\Desktop\Emotion Recognition System\haarcascade_frontalface_default.xml')
#Load our trained model in emotion_classifier variable
emotion_classifier =load_model(r'C:\Users\amit1\Desktop\Emotion Recognition System\emotion_recognition_vgg_cnn.h5')

class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

#Start the camera:  0 is for default camera and 1 is for external camera 
cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video in frame variable and get True in ret variable if it is reading a image from video.
    ret, frame = cap.read()
    
    #Convert frame from default BGR format into gray so that trained model dont get confused as it is trained on gray images. 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        #(x, y) is upper left corner in the frame
        #(x+w, y+h) is lower right corneer in the frame
        #(255,0,0) is the blue color for rectangle (B,G,R) and last argument 2 is the thickness of rectngle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class (Reason OF Interest)

            preds = emotion_classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
            
    cv2.imshow('Facial Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























