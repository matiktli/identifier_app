from .__Imports import *
from .Camera import *
from .FaceDetector import *


class PhotoCapture():

    def __init__(self, source: Video, face_detector: FaceDetector):
        self.source = source
        self.face_detector = face_detector

    def start_recording_faces_to_folder(self, path, size=(250, 250), number_of_faces_to_record=200):
        faces_counter = 0

        while (faces_counter < number_of_faces_to_record):
            frame = self.source.read_frame()

            faces_data = self.face_detector.detect_faces(frame, 1)

            self.__show_img_with_face_frame(frame, faces_data)

            if faces_data:
                face_frame = format_frame_extract_rectangle(
                    frame, faces_data[0])

                face_frame = format_frame_by_size(face_frame, size)
                print(f'({faces_counter})[FACE_DETECTED] - {path}')
                self.__save_img(face_frame, path, str(faces_counter))
                faces_counter += 1

    def __save_img(self, frame, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = path + '/' + name + '.png'
        cv2.imwrite(full_path, frame)

    def __show_img_with_face_frame(self, frame, faces_data, scale=0.5):
        tmp_frame = frame
        tmp_frame = format_frame_by_scale(frame, scale)
        if faces_data and len(faces_data[0]) > 0:
            face_data = faces_data[0]
            cv2.rectangle(
                tmp_frame, (int(face_data[0] * scale), int(face_data[1] * scale)), (int(face_data[2] * scale), int(face_data[3] * scale)), (255, 0, 0), 2)
        cv2.imshow('Face capture', tmp_frame)
        cv2.waitKey(1)
