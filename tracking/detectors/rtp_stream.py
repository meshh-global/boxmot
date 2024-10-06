import cv2
import yaml

class RTPStreamSource:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.stream_url = config['rtp_stream']['url']
        self.width = config['rtp_stream']['width']
        self.height = config['rtp_stream']['height']
        self.fps = config['rtp_stream']['fps']
        
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame
    
    def __del__(self):
        if self.cap:
            self.cap.release()

def create_rtp_source(config_path):
    return RTPStreamSource(config_path)