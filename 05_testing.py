import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

def CalculateFAR(image_folder):
    image_filenames = os.listdir(image_folder)
    false_positives = 0
    test_total = 0

    for image_filename in image_filenames:
        try:
            image_path = os.path.join(image_folder, image_filename)
            authorization = int(image_filename.split('.')[1])
            if authorization != 0: continue
            test_total += 1
        except:
             continue

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),  # Set a specific minimum size for face detection
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            assumed_auth, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence >= 40: continue
            if assumed_auth != authorization:
                #print('[-] Found false positive: {0}'.format(image_filename))
                false_positives += 1

    cv2.destroyAllWindows()
    print('false_pos: {0}, total: {1}'.format(false_positives, test_total))
    return false_positives / test_total

def CalculateFRR(image_folder):
    image_filenames = os.listdir(image_folder)
    false_negatives = 0
    test_total = 0

    for image_filename in image_filenames:
        try:
            image_path = os.path.join(image_folder, image_filename)
            authorization = int(image_filename.split('.')[1])
            if authorization != 1: continue
            test_total += 1
        except:
             continue

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),  # Set a specific minimum size for face detection
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            assumed_auth, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence >= 40: continue
            if assumed_auth != authorization:
                #print('[-] Found false negative: {0}'.format(image_filename))
                false_negatives += 1
    
    cv2.destroyAllWindows()
    print('false_neg: {0}, total: {1}'.format(false_negatives, test_total))
    return false_negatives / test_total


print("Executing the first test...")
far_rateo = CalculateFAR('test_dataset')
frr_rateo = CalculateFRR('test_dataset')
err_rateo = (far_rateo + frr_rateo) / 2
print("False Acceptance Rate (FAR) is {0}".format(far_rateo))
print("False Rejection Rate (FRR) is {0}".format(frr_rateo))
print("ERR is {0}\n\n".format(err_rateo))

print("Executing the second test...")
far_rateo2 = CalculateFAR('test_dataset2')
frr_rateo2 = CalculateFRR('test_dataset2')
err_rateo2 = (far_rateo2 + frr_rateo2) / 2
print("False Acceptance Rate (FAR) is {0}".format(far_rateo2))
print("False Rejection Rate (FRR) is {0}".format(frr_rateo2))
print("ERR is {0}".format(err_rateo2))