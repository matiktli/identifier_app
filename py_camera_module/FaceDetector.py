from .__Imports import *


class FaceDetector():

    def __init__(self, classifier_path):
        self.face_classifier = cv2.CascadeClassifier(classifier_path)

    def detect_faces(self, frame, num_faces_to_detect=1):
        faces = self.face_classifier.detectMultiScale(frame, 1.1, 4)
        if len(faces) == 0:
            return None

        result_faces_data = []
        for face in faces[:num_faces_to_detect]:
            x1, y1, w, h = face[0], face[1], face[2], face[3]
            x2, y2 = x1 + w, y1 + h
            if (x2-x1 > 30):
                result_faces_data.append((x1, y1, x2, y2))
        return result_faces_data
