from .__Imports import *


class FaceDetector():

    def __init__(self, classifier_path):
        self.face_classifier = cv2.CascadeClassifier(classifier_path)

    def detect_faces(self, frame, num_faces_to_detect=1, min_face_size=100):
        faces = self.face_classifier.detectMultiScale(frame, 1.1, 4)
        if len(faces) == 0:
            return []

        result_faces_data = []
        for face in faces[:num_faces_to_detect]:
            x1, y1, w, h = face[0], face[1], face[2], face[3]
            x2, y2 = x1 + w, y1 + h
            dx, dy = x2 - x1, y2 - y1

            face_data = (x1, y1, x2, y2, dx, dy)
            if (self.__is_face_allowed(face_data, result_faces_data)):
                result_faces_data.append(face_data)

        result_faces_data = self.__sort_faces_by_size_desc(result_faces_data)
        return result_faces_data

    def __sort_faces_by_size_desc(self, faces):
        faces.sort(key=lambda face: face[4] * face[5], reverse=True)
        return faces

    def __is_face_allowed(self, new_face, existing_faces):
        (x1, y1, x2, y2, dx, dy) = new_face

        if (dx < 100 or dy < 100):
            return False

        return True
