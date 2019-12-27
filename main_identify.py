from py_identifier_module.Identifier import CameraIdentifierStrategy, Identifier
from py_camera_module.Camera import Video
from py_camera_module.FaceDetector import FaceDetector
from py_training_module.Models import load_model
from utils.config_utils import get_config_json

RESOURCE_PATH = 'resources'
CLASSIFIER_PATH = RESOURCE_PATH + '/haarcascade_frontalface_default.xml'
MODEL_PATH = RESOURCE_PATH + '/models/' + 'model_2_2.h5'
STUDENT_IDS_CONFIG_PATH = RESOURCE_PATH + '/config/config_ids_2.JSON'


# Main func of file
def main_identifier(config_file_path=STUDENT_IDS_CONFIG_PATH, classifier_path=CLASSIFIER_PATH, model_path=MODEL_PATH):
    student_data = get_config_json(config_file_path)['students']

    video_source = Video()
    face_detector = FaceDetector(classifier_path)
    model = load_model(model_path)

    identifier = Identifier(face_detector, model, student_data)

    camera_identifier = CameraIdentifierStrategy(video_source, identifier)

    camera_identifier.start_identification()


main_identifier()
