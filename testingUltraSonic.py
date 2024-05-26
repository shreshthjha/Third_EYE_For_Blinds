import cv2
import urllib.request
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
import concurrent.futures
import requests

import pyttsx3
engine = pyttsx3.init()
#####################################

url = 'http://192.168.37.38/cam-mid.jpg'
uslink = 'http://192.168.37.38/ultrasonic'
im = None
a = 0

def waitfive(txt):
    global a
    if a == 0:
        engine.say(txt)
        engine.runAndWait()
    a += 1
    if a == 5:
        a = 0

def object_detection():
    while True:
        img_resp = urllib.request.urlopen(url)
        usresponse = requests.get(uslink)
        if usresponse.status_code == 200:
            us_val = int(usresponse.text)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
        ####################################
        bbox, label, conf = cv.detect_common_objects(frame, confidence=0.4, model='yolov4-tiny')
        out = draw_bbox(frame, bbox, label, conf)
        steps = int(us_val//30.48)-1
        if steps<0:
            steps = 0
        if label and us_val<500:
            print(label[0], steps)
            waitfive(label[0]+" is "+str(steps)+" steps away")
        elif label:
            print(label[0])
            waitfive(label[0] + " is detected")
        elif us_val>0:
            print(steps)
            waitfive(" obstacle ahead")

        cv2.imshow('object detection', out)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    # webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Object Detection Started")
    # with concurrent.futures.ProcessPoolExecutor() as executer:
    #     od = executer.submit(objectDetection)
    object_detection()