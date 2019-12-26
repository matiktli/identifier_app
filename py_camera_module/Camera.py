from .__Imports import *


class Video():

    def __init__(self, source=cv2.VideoCapture(0), color_mode=cv2.COLOR_BGR2RGB):
        self.source = source
        self.color_mode = color_mode

    def read_frame(self):
        ret, frame = self.source.read()

        if cv2.waitKey(1) and 0xFF == ord('q'):
            return None
        return frame


def format_frame_by_size(frame, size=(250, 250)):
    frame = cv2.resize(frame, size)
    return frame


def format_frame_by_scale(frame, scale=1):
    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    frame = format_frame_by_size(frame, new_size)
    return frame


def format_frame_by_color(frame, color_mode=cv2.COLOR_BGR2RGB):
    frame = cv2.cvtColor(frame, color_mode)
    return frame


def format_frame_extract_rectangle(frame, rectangle_data, buffor=50):
    x1, y1 = rectangle_data[0] - buffor, rectangle_data[1] - buffor
    x2, y2 = rectangle_data[2] + buffor, rectangle_data[3] + buffor
    return frame[y1:y2, x1:x2]
