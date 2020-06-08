# -*- coding: utf-8 -*-

import dlib
import cv2
import sys
import numpy as np


###############################################################################
#                                                                             #
#                                  DISPLAYS                                   #
#                                                                             #
###############################################################################


def drawPolyline(im, landmarks, start, end, isClosed=False):
    points = []
    for i in range(start, end+1):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(
        im, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8
        )


# Draw the lines for a 68 landmarks model
def renderFaceLines(im, landmarks, color=(0, 255, 0), radius=4):
    assert(landmarks.num_parts == 68)
    drawPolyline(im, landmarks, 0, 16)           # Jaw line
    drawPolyline(im, landmarks, 17, 21)          # Left eyebrow
    drawPolyline(im, landmarks, 22, 26)          # Right eyebrow
    drawPolyline(im, landmarks, 27, 30)          # Nose bridge
    drawPolyline(im, landmarks, 30, 35, True)    # Lower nose
    drawPolyline(im, landmarks, 36, 41, True)    # Left eye
    drawPolyline(im, landmarks, 42, 47, True)    # Right Eye
    drawPolyline(im, landmarks, 48, 59, True)    # Outer lip
    drawPolyline(im, landmarks, 60, 67, True)    # Inner lip


# Draw points for any numbers of landmarks models
def renderFacePoints(im, landmarks, color=(0, 255, 0), radius=1):
    for x, y in landmarks:
        cv2.circle(im, (x, y), radius, color, -1)


def displayText(frame, doBugEye):

    cv2.putText(
        frame,
        "BugEye 'E' ?",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    if doBugEye:
        cv2.putText(
            frame,
            "ON",
            (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(
            frame,
            "OFF",
            (175, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.putText(
        frame,
        "Quit 'ESC'",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)



###############################################################################
#                                                                             #
#                                  FUN STUFF                                  #
#                                                                             #
###############################################################################


###### SMILE

def smileScore(landmarks, threshold = 0.55):
    # Compute the size of the lips
    xLips1, yLips1 = landmarks[55]
    xLips2, yLips2 = landmarks[49]
    lipsSize = abs(xLips1 - xLips2)

    # Compute the size of the jaw
    xJaw1, yJaw1 = landmarks[11]
    xJaw2, yJaw2 = landmarks[7]
    jawSize = abs(xJaw1 - xJaw2)

    # Compute the ratio Lips / Jaw
    score = 1.0 * lipsSize / jawSize
    return score >= threshold, score


###### EYES

def barrel(src, k):
    w = src.shape[1]
    h = src.shape[0]

    # Meshgrid of destiation image
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize x and y
    x = np.float32(x) / w - 0.5
    y = np.float32(y) / h - 0.5

    # Radial distance from center
    r = np.sqrt(np.square(x) + np.square(y))

    # Implementing the following equaition
    dr =  np.multiply(k * r , np.cos(np.pi * r))

    # Outside the maximum radius dr is set to 0
    dr[r > 0.5] = 0

    # Inverse mapping to remap
    rn = r - dr

    # Applying the distortion on the grid
    xd = cv2.divide(np.multiply(rn, x), r)
    yd = cv2.divide(np.multiply(rn, y), r)

    # Back to un-normalized coordinates
    xd = w * (xd + 0.5)
    yd = h * (yd + 0.5)

    return cv2.remap(src, xd, yd, cv2.INTER_CUBIC)


def bugEye(src, landmarks, radius = 30, bulgeAmount = 0.75):
    # Find the roi for left Eye
    roiEyeLeft = [ landmarks[37][0] - radius, landmarks[37][1] - radius,
          (landmarks[40][0] - landmarks[37][0] + 2*radius),
          (landmarks[41][1] - landmarks[37][1] + 2*radius)  ]
    # Find the roi for right Eye
    roiEyeRight = [ landmarks[43][0] - radius, landmarks[43][1] - radius,
          (landmarks[46][0] - landmarks[43][0] + 2*radius),
          (landmarks[47][1] - landmarks[43][1] + 2*radius)  ]

    # Find the patch for left eye and apply the transformation
    leftEyeRegion = src[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3],
                        roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]]
    leftEyeRegionDistorted = barrel(leftEyeRegion, bulgeAmount)

    # Find the patch for right eye and apply the transformation
    rightEyeRegion = src[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3],
                         roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]]
    rightEyeRegionDistorted = barrel(rightEyeRegion, bulgeAmount)

    output = np.copy(src)
    output[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3],
           roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]] = leftEyeRegionDistorted
    output[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3],
           roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]] = rightEyeRegionDistorted

    return output


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


def main():
    # Landmark model location
    PREDICTOR_68_PATH = "./data/models/shape_predictor_68_face_landmarks.dat"
    # Get the face detector
    faceDetector = dlib.get_frontal_face_detector()
    # The landmark detector is implemented in the shape_predictor class
    landmarkDetector_68 = dlib.shape_predictor(PREDICTOR_68_PATH)

    # Create a VideoCapture object
    video = cv2.VideoCapture(0)

    doBugEye = False

    try:
        while(True):
            # Grab a frame
            ret, frame = video.read()
            if not ret:
                break

            # BGR -> RGB
            imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = faceDetector(imDlib, 0)

            # Display some text
            displayText(frame, doBugEye)

            # For every detected faces
            for faceRect in faces:
                # Get the facial landmarks
                landmarks = [
                        (p.x, p.y) for p in landmarkDetector_68(
                                imDlib, faceRect).parts()
                        ]

                # Get the surround box and draw it on the frame
                top, right, bottom, left = (
                        faceRect.top(),
                        faceRect.right(),
                        faceRect.bottom(),
                        faceRect.left())
                cv2.rectangle(
                    frame,
                    (left, top), (right, bottom),
                    (0, 0, 255), 2)
                cv2.rectangle(
                    frame,
                    (left, bottom - 20), (right, bottom),
                    (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX

                # Detect if the person smiling
                isSmiling, valueSmile = smileScore(landmarks)
                if isSmiling:
                    cv2.putText(
                        frame, "Smiling :)",
                        (left + 3, bottom - 3),
                        font, 0.75, (255, 255, 255), 1)
                else:
                    cv2.putText(
                        frame, "Not :/",
                        (left + 3, bottom - 3),
                        font, 0.75, (255, 255, 255), 1)

                if doBugEye:
                    frame = bugEye(frame, landmarks)

            # Display the image
            cv2.imshow("Fun face app", frame)

            # Key event
            key = cv2.waitKey(1)
            if key == 27:                             # ESC - quit the app
                break
            if key == ord('e') or key == ord('E'):    # E - eyes
                doBugEye = not doBugEye

    except:
        print("Something went wrong:", sys.exc_info())

    # Destroy the windows
    cv2.destroyAllWindows()
    # Release the webcam
    video.release()

if __name__ == '__main__':
    main()
