import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame. Video streaming is discrete, not continuous
    ret, img = cap.read()
    
  # Detect the faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = cv2.putText(img, 'Face Detected', (img.shape[0]+3, img.shape[0]), font, 2, (255, 255, 255), 2)

        eyes = eyes_cascade.detectMultiScale(img, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = cv2.putText(img, 'Eyes Detected', (0, img.shape[0]), font, 2, (255, 255, 255), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()

