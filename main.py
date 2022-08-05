from select import KQ_NOTE_LOWAT
import cv2
import face_recognition
import numpy as np

capture = cv2.VideoCapture(0)


kd = face_recognition.load_image_file('images/kd.jpeg')
kd_encoded = face_recognition.face_encodings(kd)[0]

dame = face_recognition.load_image_file('images/dame.jpeg')
dame_encoded = face_recognition.face_encodings(dame)[0]

lebron = face_recognition.load_image_file('images/lebron.webp')
lebron_encoded = face_recognition.face_encodings(lebron)[0]

curry = face_recognition.load_image_file('images/curry.jpeg')
curry_encoded = face_recognition.face_encodings(curry)[0]

pg = face_recognition.load_image_file('images/pg.jpeg')
pg_encoded = face_recognition.face_encodings(pg)[0]

klaw = face_recognition.load_image_file('images/klaw.jpeg')
klaw_encoded = face_recognition.face_encodings(klaw)[0]

kobe = face_recognition.load_image_file('images/kobe.jpeg')
kobe_encoded = face_recognition.face_encodings(kobe)[0]

osi = face_recognition.load_image_file('images/Osi.jpeg')
osi_encoded = face_recognition.face_encodings(osi)[0]

kyrie = face_recognition.load_image_file('images/kyrie.jpeg')
kyrie_encoded = face_recognition.face_encodings(kyrie)[0]

harden = face_recognition.load_image_file('images/harden.jpeg')
harden_encoded = face_recognition.face_encodings(harden)[0]

giannis = face_recognition.load_image_file('images/giannis.jpg')
giannis_encoded = face_recognition.face_encodings(giannis)[0]

known_face_encodings = [
    kd_encoded,
    dame_encoded,
    lebron_encoded,
    curry_encoded,
    pg_encoded,
    klaw_encoded,
    kobe_encoded,
    osi_encoded,
    kyrie_encoded,
    harden_encoded,
    giannis_encoded
]

known_faces = [
    'KD',
    'Dame',
    'Lebron',
    'Curry',
    'PG',
    'Klaw',
    'Kobe',
    'Osi',
    'Kyrie',
    'Harden',
    'Giannis'
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_faces[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows() 



