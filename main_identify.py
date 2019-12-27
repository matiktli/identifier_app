from py_identifier.Identifier import CameraIdentifierStrategy, Identifier
from py_camera_module.Camera import Video
from py_camera_module.FaceDetector import FaceDetector
from py_training_module.Models import load_model
from utils.config_utils import get_config_json

RESOURCE_PATH = 'resources'
CLASSIFIER_PATH = RESOURCE_PATH + '/haarcascade_frontalface_default.xml'
MODEL_PATH = RESOURCE_PATH + '/models/' + 'model_1.h5'
STUDENT_IDS_CONFIG_PATH = RESOURCE_PATH + '/config/config_ids.JSON'

student_data = get_config_json(STUDENT_IDS_CONFIG_PATH)['students']

video_source = Video()
face_detector = FaceDetector(CLASSIFIER_PATH)
model = load_model(MODEL_PATH)

identifier = Identifier(face_detector, model, student_data)

camera_identifier = CameraIdentifierStrategy(video_source, identifier)

camera_identifier.start_identification()
