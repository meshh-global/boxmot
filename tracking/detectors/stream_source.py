import cv2
import yaml
import socket
import numpy as np
import netifaces as ni

class StreamSource:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.stream_type = config['stream']['type']
        self.stream_url = config['stream']['url']
        self.width = config['stream']['width']
        self.height = config['stream']['height']
        self.fps = config['stream']['fps']
        
        if self.stream_type == 'rtp':
            self.cap = cv2.VideoCapture(self.stream_url)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        elif self.stream_type == 'udp':
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ni.ifaddresses('ens3')
            ip = ni.ifaddresses('ens3')[ni.AF_INET][0]['addr']
            print(ip)
            proto, host, port = self.stream_url.split(':')
            print(proto, host, port)
            self.sock.bind((ip, int(port)))
            self.buffer_size = self.width * self.height * 3  # Assuming 3 channels (RGB)
        else:
            raise ValueError(f"Unsupported stream type: {self.stream_type}")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.stream_type == 'rtp':
            ret, frame = self.cap.read()
            if not ret:
                raise StopIteration
            return frame
        elif self.stream_type == 'udp':
            data, _ = self.sock.recvfrom(self.buffer_size)
            frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
            return frame
    
    def __del__(self):
        if self.stream_type == 'rtp' and self.cap:
            self.cap.release()
        elif self.stream_type == 'udp' and self.sock:
            self.sock.close()

def create_stream_source(config_path):
    return StreamSource(config_path)