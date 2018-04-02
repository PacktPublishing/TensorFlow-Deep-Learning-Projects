from tensorflow_detection import DetectionObj

if __name__ == "__main__":
    detection = DetectionObj(model='ssd_mobilenet_v1_coco_11_06_2017')

    detection.webcam_pipeline()
