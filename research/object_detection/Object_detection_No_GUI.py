######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import serial
import time


# Set up camera constants
IM_WIDTH = 1280
IM_HEIGHT = 720
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
#camera_type = 'picamera'
#parser = argparse.ArgumentParser()
#parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
#                    action='store_true')
#args = parser.parse_args()
#if args.usbcam:
camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'
MODEL_NAME = 'mask_rcnn_resnet101_atrous_coco_2018_01_28'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# aaa
# GUI
show_gui = False
# Serial
send_to_serial = False
if send_to_serial:
    ser = serial.Serial('/dev/tty.usbmodemFD121', 9600)
    empty = bytes.fromhex("3c010101010101010101013e")
    obj1 = bytes.fromhex("3cFFFF01010101010101013e")
    obj2 = bytes.fromhex("3c0101FFFF0101010101013e")
    obj3 = bytes.fromhex("3c01010101FFFF010101013e")
    obj4 = bytes.fromhex("3c010101010101FFFF01013e")
    # empty = '<' + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + '>'
    # obj1 = '<' + chr(255) + chr(255) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + '>'
    # obj2 = '<' + chr(1) + chr(1) + chr(255) + chr(255) + chr(1) + chr(1) + chr(1) + chr(1) + '>'
    # obj3 = '<' + chr(1) + chr(1) + chr(1) + chr(1) + chr(255) + chr(255) + chr(1) + chr(1) + '>'
    # obj4 = '<' + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(1) + chr(255) + chr(255) + '>'

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = frame1.array
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    count = 0
    sum_fps = 0

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        if show_gui:
            vis_util.visualize_boxes_and_labels_on_image_array(
               frame,
               np.squeeze(boxes),
               np.squeeze(classes).astype(np.int32),
               np.squeeze(scores),
               category_index,
               use_normalized_coordinates=True,
               line_thickness=8,
               min_score_thresh=0.85)

        detected = [i for i in enumerate(np.squeeze(scores)) if i[1] >= 0.85]

        os.system('cls' if os.name == 'nt' else 'clear')
        print("Detected objects:")
        for item in detected:
            # Class
            id_class = np.squeeze(classes)[item[0]]
            print("{0}, {1:.2f}%".format(category_index[id_class], item[1]*100))

            # Positions of bbox
            box = np.squeeze(boxes)[item[0]]
            top = box[0]
            left = box[1]
            bottom = box[2]
            right = box[3]
            print("box: {}".format(box))

            if send_to_serial:
                # Map position to vest
                rows = 4
                cols = 2
                vest_activation_position = np.ones([rows, cols])
                position = "3c" # FFFF01010101010101013e
                for row in range(0, rows):
                    for col in range(0, cols):
                        q_x1 = (1 / cols) * col;
                        q_y1 = (1 / rows) * row;
                        q_x2 = (1 / cols) * (col + 1);
                        q_y2 = (1 / rows) * (row + 1);

                        if q_x1 > left:
                            x1 = q_x1;
                        else:
                            x1 = left;

                        if q_y1 > top:
                            y1 = q_y1;
                        else:
                            y1 = top;

                        if q_x2 > right:
                            x2 = right;
                        else:
                            x2 = q_x2;

                        if q_y2 > bottom:
                            y2 = bottom;
                        else:
                            y2 = q_y2;

                        # percent of the detected object in the quadrant
                        image_area_in_quadrant = (x2 - x1) * (y2 - y1)
                        image_area = (right - left) * (bottom - top)
                        if image_area_in_quadrant > 0:
                            value = (image_area_in_quadrant / image_area) * 255.0
                            if int(value) == 0:
                                value = 1
                            vest_activation_position[row][col] = value
                        position = position + str(hex(int(vest_activation_position[row][col]))).replace('0x','')
                position = position + "3e"

                # write class to serial
                # bottle
                if id_class == 44:
                    print("Writing obj to serial: {}".format(obj1))
                    ser.write(obj1)
                # cup
                if id_class == 47:
                    print("Writing obj to serial: {}".format(obj2))
                    ser.write(obj2)
                # scissors
                if id_class == 87:
                    print("Writing obj to serial: {}".format(obj3))
                    ser.write(obj3)
                # keyboard
                if id_class == 76:
                    print("Writing obj to serial: {}".format(obj4))
                    ser.write(obj4)

                # write position to serial
                if id_class == 44 or id_class == 47 or id_class == 87 or id_class == 76:
                    time.sleep(0.2)
                    ser.write(empty)
                    time.sleep(0.2)
                    print("Writing position to serial: {}".format(bytes.fromhex(position)))
                    ser.write(bytes.fromhex(position))
                    time.sleep(0.2)
                    ser.write(empty)
                    time.sleep(0.2)

        if show_gui:
            cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        print("FPS: {0:.2f}".format(frame_rate_calc))

        # All the results have been drawn on the frame, so it's time to display it.
        if show_gui:
            cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        count = count + 1
        sum_fps = sum_fps + frame_rate_calc

        if count == 30:
            break

        # Press 'q' to quit
        if show_gui:
            if cv2.waitKey(1) == ord('q'):
                break

    camera.release()

    print("FPS Avg: {0:.2f}".format(sum_fps/count))

if send_to_serial:
    ser.close()
if show_gui:
    cv2.destroyAllWindows()

