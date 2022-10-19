import cv2
import face_recognition
import os
import numpy as np

cap = cv2.VideoCapture(0)
file_names = os.listdir("attendance folder")
images = []
encodes = []
for file_name in file_names:
    image_path = os.path.join("attendance folder", file_name)
    ex_image = cv2.imread(image_path)
    images.append(ex_image)
    encodes.append(face_recognition.face_encodings(ex_image)[0])


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    distances = []
    cam_locs = face_recognition.face_locations(frame)
    if len(cam_locs) != 0:
        cam_encode = face_recognition.face_encodings(frame, cam_locs)[0]
        for encode in encodes:
            dist = face_recognition.face_distance([encode], cam_encode)
            distances.append(dist)

        image_index = np.argmin(distances)
        cv2.putText(frame, f"{file_names[image_index]}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2)

    





    cv2.imshow("camera", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()