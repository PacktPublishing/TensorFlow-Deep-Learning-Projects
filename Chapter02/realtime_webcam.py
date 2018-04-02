from tensorflow_detection import DetectionObj
from threading import Thread
import cv2

def resize(image, new_width=None, new_height=None):
    """
    Resize an image based on a new width or new height
    keeping the original ratio
    """
    height, width, depth = image.shape
    if new_width:
        new_height = int((new_width / float(width)) * height)
    elif new_height:
        new_width = int((new_height / float(height)) * width)
    else:
        return image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

class WebcamStream:
    def __init__(self):
        # Initialize Webcam
        self.stream = cv2.VideoCapture(0)
        # Starting TensorFlow API with SSD Mobilenet
        self.detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')
        # Start capturing video so the Webca, will tune itself
        _, self.frame = self.stream.read()
        # Set the stop flag to False
        self.stop = False
        #
        Thread(target=self.refresh, args=()).start()

    def refresh(self):
        # Looping until an explicit stop is sent
        # from outside the function
        while True:
            if self.stop:
                return
            _, self.frame = self.stream.read()

    def get(self):
        # returning the annotated image
        return self.detection.annotate_photogram(self.frame)

    def halt(self):
        # setting the halt flag
        self.stop = True

if __name__ == "__main__":
    stream = WebcamStream()

    while True:
        # Grabbing the frame from the threaded video stream
        # and resize it to have a maximum width of 400 pixels
        frame = resize(stream.get(), new_width=400)
        cv2.imshow("Webcam", frame)
        # If the space bar is hit, the program will stop
        if cv2.waitKey(1) & 0xFF == ord(" "):
            # First stopping the streaming thread
            stream.halt()
            # Then halting the while loop
            break

