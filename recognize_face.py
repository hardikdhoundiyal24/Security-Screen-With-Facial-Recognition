import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

valid_face_id = 1

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    valid_face_detected = False

    for (x, y, w, h) in faces:
        
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

   
        if id == valid_face_id and confidence < 70:  
            cv2.putText(frame, "Valid Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            valid_face_detected = True
        else:
           
            valid_face_detected = False

  
    if not valid_face_detected and len(faces) > 0:
       
        black_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.namedWindow('Black Screen', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Black Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            cv2.imshow('Black Screen', black_screen)
            print("Unrecognized face detected. Press 'e' to exit the black screen.")

            
            if cv2.waitKey(1) & 0xFF == ord('e'):
                cv2.destroyWindow('Black Screen')
                break


    cv2.imshow('Camera', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
