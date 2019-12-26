from py_camera_module.VideoCapture import *
from py_camera_module.Camera import Video
from py_camera_module.FaceDetector import FaceDetector

RESOURCE_PATH = 'resources'
CLASIFIER_PATH = RESOURCE_PATH + '/haarcascade_frontalface_default.xml'
FACE_DATA_PATH = RESOURCE_PATH + '/face_data/raw'

studentName = input('Provide student name:\n')
studentName = studentName.replace(' ', '')

numberOfFaces = input('How many faces to collect:\n')
numberOfFaces = int(numberOfFaces)

videoSource = Video()
faceDetector = FaceDetector(CLASIFIER_PATH)
faceCollector = PhotoCapture(videoSource, faceDetector)

path_to_save = FACE_DATA_PATH + '/' + studentName
input('[FACE DATA WILL BE COLLECTED] - press any key to start...')

faceCollector.start_recording_faces_to_folder(
    path=path_to_save, number_of_faces_to_record=numberOfFaces)
