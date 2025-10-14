from ImageCapture import ImageCapture

class Preprocessing:
    def __init__(self, camera_index="/dev/video0"):
        self.image_capture = ImageCapture(camera_index)

    def get_image(self):
        return self.image_capture.capture_image()

    def release_camera(self):
        self.image_capture.release()
