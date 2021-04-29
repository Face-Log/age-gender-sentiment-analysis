import cv2
import numpy as np
import dlib
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array

cap = cv2.VideoCapture(0)

model_emo = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
model_emo.load_weights('models/facial_expression_model_weights.h5')

model_gen = load_model("models/gender_detection.model")

age_net = cv2.dnn.readNetFromCaffe('models/deployage.prototxt', 'models/age_net.caffemodel')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
genders = ('Male','Female')
ages = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)', '(48-53)', '(60-100)']

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        detected_face0 = frame[y1:y2,x1:x2]

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_face = cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        detected_face1 = frame[int(y1):int(y2) , int(x1):int(x2)]
        detected_face2 = cv2.cvtColor(detected_face1, cv2.COLOR_BGR2GRAY)
        detected_face3 = cv2.resize(detected_face2, (48, 48))
        img_pixels = detected_face3
        img_pixels = img_to_array(img_pixels)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        face_crop = cv2.resize(detected_face1, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        blob = cv2.dnn.blobFromImage(detected_face1, 1, (227, 227), MODEL_MEAN_VALUES,swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = ages[age_preds[0].argmax()]

        predictions_emo = model_emo.predict(img_pixels)

        predictions_gen = model_gen.predict(face_crop)[0]

        max_index_emo = np.argmax(predictions_emo[0])

        emotion = emotions[max_index_emo]

        idx = np.argmax(predictions_gen)
        label = genders[idx]
        label = "{}: {:.2f}%".format(label, predictions_gen[idx] * 100)

        overlay_text = "%s  %s  %s" % (label, age, emotion)

        cv2.putText(frame, overlay_text , (x1+50,y2+50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()