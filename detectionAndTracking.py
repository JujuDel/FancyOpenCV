# -*- coding: utf-8 -*-

import numpy as np
import cv2


# Colors
g_colorDetection = (255, 0, 0)  # BLUE
g_colorTracking = (0, 255, 0)  # GREEN

# Tresholds
g_nbFramesDetection = 5
g_nbFramesTracking = 10

# Input res for YOLO net
# 320
# 416
# 609
g_resYOLO = (320, 320)


#####################################################
#                                                   #
#                     DETECTION                     #
#                                                   #
#####################################################


# Load Yolo
def loadYOLOv3():
    # Network
    net = cv2.dnn.readNet("data/yolo/yolov3.weights", "data/yolo/yolov3.cfg")
    # Layers
    layer_names = net.getLayerNames()
    # Output Layers
    output_layers = [
            layer_names[i[0] - 1]
            for i in net.getUnconnectedOutLayers()]
    # Classes
    classes = []
    with open("data/yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, classes, layer_names, output_layers


# Detect the objects
def detectObjects(net, output_layers, frame):
    # Resize the image
    img = cv2.resize(frame, g_resYOLO)
    # Pass the input in the net
    blob = cv2.dnn.blobFromImage(
            img, 0.00392, g_resYOLO, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Run the net
    outs = net.forward(output_layers)
    return outs


def computeBoxes(outs, height, width):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids


def display(frame, boxes, indexes, classes, class_ids, color):
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            if (label == "sports ball"):
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                return True, boxes[i]
    return False, None


def applyDetection(frame, net, classes, output_layers):
    height, width, channels = frame.shape

    # Detecting objects
    outs = detectObjects(net, output_layers, frame)

    # Compute the boxes
    boxes, confidences, class_ids = computeBoxes(outs, height, width)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

    # Return weither or not a box for a ball is found and display it
    return display(frame, boxes, indexes, classes, class_ids, g_colorDetection)


#####################################################
#                                                   #
#                     TRACKING                      #
#                                                   #
#####################################################


def createTracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    if tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    if tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    return None


#####################################################
#                                                   #
#                       TEXT                        #
#                                                   #
#####################################################


def addText(frame, isPlay, isWaitonNextObjDetected):

    cv2.putText(
        frame,
        "Status:",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Pause on Detection ('W'):",
        (900, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    if (isWaitonNextObjDetected):
        cv2.putText(
            frame,
            "ON",
            (1212, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(
            frame,
            "OFF",
            (1212, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(
        frame,
        "Pause ('P')",
        (1062, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Force Detection on next frame ('D')",
        (764, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Replay ('R')",
        (1058, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Exit ('Q' or 'Esc')",
        (990, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    cv2.putText(
        frame,
        "Nb frame skipped on Detection failure:",
        (735, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(
        frame,
        str(g_nbFramesDetection),
        (1212, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Max nb frame on Tracking failure before detection:",
        (588, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(
        frame,
        str(g_nbFramesTracking),
        (1212, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


#####################################################
#                                                   #
#                       MAIN                        #
#                                                   #
#####################################################
g_isWaitonNextObjDetected = False  # Global for replay case


def main():
    global g_isWaitonNextObjDetected, g_colorDetection, g_colorTracking

    # Open the video
    cap = cv2.VideoCapture('data/videos/soccer-ball.mp4')

    if not cap.isOpened():
        print("Error opening video stream or file")

    # Load YOLO
    net, classes, layer_names, output_layers = loadYOLOv3()

    # Frame counter
    countFramePassed = 0

    # Booleans
    isObjDetected = False
    isPlay = True
    isReplay = False

    # Tracker KCF
    trackerKCF = createTracker("TLD")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outVid = cv2.VideoWriter("data/results/output.mp4", fourcc, 20.0, (1280, 720))

    # Read until video is completed
    while(cap.isOpened()):
        if (isPlay):

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                if not(isObjDetected):
                    if countFramePassed == 0:
                        isFound, bbox = applyDetection(
                                frame, net, classes, output_layers)
                        if isFound:
                            cv2.putText(
                                frame,
                                "Detection OK",
                                (60, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                            # Initialize tracker: first frame and bounding box
                            trackerKCF = createTracker("KCF")
                            ret = trackerKCF.init(frame, bbox)
                            # Set params
                            isObjDetected = True
                            countFramePassed = 0
                            if (g_isWaitonNextObjDetected):
                                isPlay = False
                        else:
                            countFramePassed += 1
                            cv2.putText(
                                frame,
                                "Detection failure, " +
                                str(countFramePassed - 1) +
                                " / " + str(g_nbFramesDetection) +
                                " frames skiped",
                                (60, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    else:

                        if countFramePassed > g_nbFramesDetection:
                            countFramePassed = 0
                        else:
                            countFramePassed += 1
                        cv2.putText(
                            frame,
                            "Detection failure, " +
                            str(countFramePassed - 1) +
                            " / " + str(g_nbFramesDetection) +
                            " frames skiped",
                            (60, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    # Update tracker
                    ret, bbox = trackerKCF.update(frame)
                    if ret:
                        # Tracking success
                        cv2.putText(
                            frame,
                            "Tracking OK",
                            (60, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, g_colorTracking, 2, 1)
                        countFramePassed = 0
                    elif countFramePassed > g_nbFramesTracking:
                        # Tracking failure for a 'long' time
                        isObjDetected = False
                        countFramePassed = 0
                    else:
                        # Tracking failure for a 'short' time
                        cv2.putText(
                                frame,
                                "Tracking failure detected for " +
                                str(countFramePassed) +
                                " frames",
                                (60, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        countFramePassed += 1

                addText(frame, isPlay, g_isWaitonNextObjDetected)

                k = cv2.waitKey(25)
                if k == ord('q') or k == ord('Q') or k == 27:  # Quit
                    break
                elif k == ord('p') or k == ord('P'):  # Pause
                    isPlay = not(isPlay)
                elif k == ord('d') or k == ord('D'):  # Detection on next frame
                    isObjDetected = False
                    countFramePassed = 0
                elif k == ord('w') or k == ord('W'):  # On next detection, wait
                    g_isWaitonNextObjDetected = not(g_isWaitonNextObjDetected)
                elif k == ord('r') or k == ord('R'):  # On next detection, wait
                    isReplay = True
                    break

                if not(isPlay):
                    cv2.putText(
                        frame,
                        "Paused",
                        (60, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                else:
                    cv2.putText(
                        frame,
                        "Play",
                        (60, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Display the resulting frame
                outVid.write(frame)
                cv2.imshow('Frame', frame)

            # Break the loop
            else:
                break
        else:
            k = cv2.waitKey(25)
            if k == ord('q') or k == ord('Q') or k == 27:  # Quit
                break
            elif k == ord('p') or k == ord('P'):  # Pause
                isPlay = not(isPlay)
            elif k == ord('d') or k == ord('D'):  # Detection on next frame
                isObjDetected = False
                countFramePassed = 0
            elif k == ord('w') or k == ord('W'):  # On next detection, wait
                g_isWaitonNextObjDetected = not(g_isWaitonNextObjDetected)
            elif k == ord('r') or k == ord('R'):  # On next detection, wait
                isReplay = True
                break

    # When everything done, release the video capture object
    cap.release()
    outVid.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    if isReplay:
        main()


if __name__ == "__main__":
    main()
