from py_camera_module.Camera import Video, format_frame_by_size, format_frame_extract_rectangle, format_frame_by_scale
from py_camera_module.FaceDetector import FaceDetector
import tensorflow as tf
import numpy as np
import cv2


class Identifier():

    def __init__(self, face_detector, prediction_model, student_data=None):
        self.face_detector = face_detector
        self.predition_model = prediction_model
        self.student_data = student_data

    def predict_student_from_frame(self, frame, faces_to_predict=5):
        faces_data = self.face_detector.detect_faces(frame, faces_to_predict)
        if faces_data == None:
            return [(- 1, [], 0)]

        result = []
        for face_data in faces_data:
            valid_frame = self.__extract_valid_frame_for_prediction(
                frame, face_data)
            predicted_student_id, confidence = self.__make_prediction_for_student_id(
                valid_frame)
            result.append((predicted_student_id, face_data, confidence))
        return result

    def __extract_valid_frame_for_prediction(self, frame, face_data):
        try:
            frame = format_frame_extract_rectangle(frame, face_data)
            frame = format_frame_by_size(frame, (250, 250))
            return frame
        except Exception as ex:
            print(str(ex))
            return np.zeros((250, 250, 3))

    def __make_prediction_for_student_id(self, valid_frame):
        valid_frame = tf.expand_dims(valid_frame, axis=0)
        valid_frame = tf.cast(valid_frame, tf.float32)
        predictions = self.predition_model.predict(valid_frame)
        confidence = np.amax(predictions)
        predicted_student_id = np.where(
            predictions == confidence)[1][0]
        return predicted_student_id, confidence

    def attach_info_to_frame(self, frame, student_id, face_data, confidence, i=0):
        student_name = self.__get_student_name(student_id)
        text = f'({str(i)})_ {student_name} : {str(confidence)}'
        cv2.rectangle(
            frame, (face_data[0], face_data[1]), (face_data[2], face_data[3]), (255, 0, 0), 2)
        cv2.putText(frame, text, (face_data[0] + 20, face_data[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f'{text}')
        return frame

    def __get_student_name(self, student_id):
        if student_id == -1:
            return '<UNKNOWN>'
        for student in self.student_data:
            if int(student['id']) == int(student_id):
                return student['name']
        return None


class CameraIdentifierStrategy():

    def __init__(self, video_source, indentifier):
        self.identifier = indentifier
        self.video_source = video_source

    def start_identification(self):
        while(True):
            frame = self.video_source.read_frame()
            data = self.identifier.predict_student_from_frame(
                frame)
            for i, (student_id, face_data, confidence) in enumerate(data):
                if student_id != -1:
                    frame = self.identifier.attach_info_to_frame(
                        frame, student_id, face_data, confidence, i)
            frame = format_frame_by_scale(frame, 0.7)
            cv2.imshow('I CAN SEE YOU', frame)
