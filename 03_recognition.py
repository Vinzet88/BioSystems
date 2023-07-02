import cv2
import numpy as np
import os 



def ResolveNameFromId(id, image_folder):
    image_filenames = os.listdir(image_folder)
    for image_filename in image_filenames:
        identifier = int(image_filename.split('.')[2])
        if(id == identifier): 
            name = str(image_filename.split('.')[0])
            return name 
    return 'Unknown'


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

auth_level = ['Unauthorized', 'Authorized'] 
auth_db = {}
image_folder = 'dataset'
image_filenames = os.listdir(image_folder)
for image_filename in image_filenames:
    authorization = int(image_filename.split('.')[1])
    identifier = int(image_filename.split('.')[2])
    if(identifier in auth_db): continue
    auth_db[identifier] = authorization

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Check if confidence is less them 40 ==> "0" is perfect match 
        if (confidence < 60):
            subject_auth = auth_db[id]
            confidence = "  {0}%".format(round(100 - confidence))
            subject_auth_level = auth_level[subject_auth]
            name = ResolveNameFromId(id, image_folder)
            cv2.putText(img, str(subject_auth_level), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            cv2.putText(img, str(name), (x+5,y-50), font, 1, (255,255,255), 2)

        else:
            subject_auth = 0
            confidence = "  {0}%".format(round(100 - confidence))
            subject_auth_level = auth_level[subject_auth]
            cv2.putText(img, str(subject_auth_level), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            cv2.putText(img, str('Unknown'), (x+5,y-50), font, 1, (255,255,255), 2)    
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
