import cv2
import numpy as np
import face_recognition

img_andrei = face_recognition.load_image_file("images/Andrei.PNG")
img_andrei = cv2.cvtColor(img_andrei, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file("images/rand.png")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img_andrei)[0] #[0] because face_locations() returns an array. [0] gets tuple inside
encodeAndrei = face_recognition.face_encodings(img_andrei)[0]
cv2.rectangle(img_andrei,(faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,0,255), 2)

faceLocTest = face_recognition.face_locations(img_test)
print(faceLocTest)
encodeTest = face_recognition.face_encodings(img_test)[2]
cv2.rectangle(img_test,(faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0,0,255), 2)

results = face_recognition.compare_faces([encodeAndrei],encodeTest)
faceDist = face_recognition.face_distance([encodeAndrei],encodeTest)
print(results, faceDist)

cv2.putText(img_test, f'{results} {round(faceDist[0],2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow("Image of Andrei", img_andrei)
cv2.imshow("TEST IMAGE", img_test)
cv2.waitKey()