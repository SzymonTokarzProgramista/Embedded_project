import cv2

class ImageCapture:
    def __init__(self, camera_index="/dev/video0"):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")
        return frame

    def release(self):
        self.cap.release()