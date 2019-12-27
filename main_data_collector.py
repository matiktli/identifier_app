from py_camera_module.VideoCapture import *
from py_camera_module.Camera import Video
from py_camera_module.FaceDetector import FaceDetector

RESOURCE_PATH = 'resources'
CLASIFIER_PATH = RESOURCE_PATH + '/haarcascade_frontalface_default.xml'
FACE_DATA_PATH = RESOURCE_PATH + '/face_data/raw'


# Main func of file
def main_data_collector(raw_folder_path=FACE_DATA_PATH, classifier_path=CLASIFIER_PATH):
    studentName = input('Provide student name:\n')
    studentName = studentName.replace(' ', '')

    numberOfFaces = input('How many faces to collect:\n')
    numberOfFaces = int(numberOfFaces)

    videoSource = Video()
    faceDetector = FaceDetector(classifier_path)
    faceCollector = PhotoCapture(videoSource, faceDetector)

    path_to_save = raw_folder_path + '/' + studentName
    input('[FACE DATA WILL BE COLLECTED] - press any key to start...')

    faceCollector.start_recording_faces_to_folder(
        path=path_to_save, number_of_faces_to_record=numberOfFaces)


main_data_collector()