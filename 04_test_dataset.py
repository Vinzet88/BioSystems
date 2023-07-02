import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480) 

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

surname = input('\n [!] Input the last name of the user > ')
authorization = int(input('\n [!] Input 1 for authorized, 0 for unauthorized. Any other value will be considered as unauthorized > '))

# Make sure the identifier is unique per person
identifier = int(input('\n [!] Input an incremental identifier (MAKE SURE IT\'S UNIQUE!) > '))

if(authorization != 1 and authorization != 0):
    authorization = 0

print("\n [*] Initializing face capture. Look the camera and wait...")
count = 0

while(count != 90):
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite("test_dataset/" + str(surname) + '.' + str(authorization) + '.' + str(identifier) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imwrite("test_dataset2/" + str(surname) + '.' + str(authorization) + '.' + str(identifier) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    # check for pressing of the ESC key to break the loop
    k = cv2.waitKey(100) & 0xff 
    if k == 27:
        break

print("\n [+] Dataset gathering for the training successful, now run 02_training.py to train the AI model.")
cam.release()
cv2.destroyAllWindows()