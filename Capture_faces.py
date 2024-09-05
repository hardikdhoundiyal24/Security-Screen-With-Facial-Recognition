import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_id = 1 
count = 0

print("Look at the camera and press 'c' to capture samples of your face.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        
        count += 1
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
    
    cv2.imshow('Capture Your Face', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    if count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
print("Face samples collected.")
