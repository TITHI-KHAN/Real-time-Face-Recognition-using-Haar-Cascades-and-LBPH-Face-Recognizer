import cv2
import os
import numpy as np


training_data_path = 'training_data'


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = [] 
    label_map = {}
    label = 0

    

    for dir_name in dirs:
        subject_path = os.path.join(data_folder_path, dir_name)
        if not os.path.isdir(subject_path):
            continue

        label_map[label] = dir_name

        for image_name in os.listdir(subject_path):
            if image_name.startswith("."):
                continue  # Ignore system files

            image_path = os.path.join(subject_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            faces_rect = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=10)
            if len(faces_rect) != 1:
                continue  # Use images with a single detected face

            (x, y, w, h) = faces_rect[0]  # Extract face coordinates
            face = image[y:y+h, x:x+w]  # Crop face
            faces.append(face)
            labels.append(label)

        label += 1

    return faces, labels, label_map


faces, labels, label_map = prepare_training_data(training_data_path)
face_recognizer.train(faces, np.array(labels))


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces_rect = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces_rect:
        roi_gray = gray_frame[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        label_text = label_map.get(label, "Unknown")
        cv2.putText(frame, label_text, (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
