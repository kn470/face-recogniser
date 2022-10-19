import cv2
import face_recognition

imgElon = cv2.imread("elonmusk.jpeg")
imgTest = cv2.imread("elontest.jpeg")

face_loc1 = face_recognition.face_locations(imgElon)[0]
face_loc2 = face_recognition.face_locations(imgTest)[0]

encode_origin = face_recognition.face_encodings(imgElon)[0]
encode_test = face_recognition.face_encodings(imgTest)[0]

result = face_recognition.compare_faces([encode_origin], encode_test)
face_dist = face_recognition.face_distance([encode_origin], encode_test)

cv2.rectangle(imgElon, (face_loc1[3], face_loc1[0]), (face_loc1[1], face_loc1[2]), (255,0,0), 1)
cv2.rectangle(imgTest, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]), (255,0,0), 1)
print(result)
print(face_dist)

cv2.imshow("original", imgElon)
cv2.imshow("test", imgTest)
cv2.waitKey(0)