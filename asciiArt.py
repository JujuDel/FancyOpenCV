# -*- coding: utf-8 -*-

import cv2
import numpy as np


# Max width in the console display
_MAX_WIDTH = 80
# Max height in the console display
_MAX_HEIGHT = 40

# ASCII tables for the display
_CHAR_ASCII_10 = ".:-=+*#%@"
_CHAR_ASCII_70 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# Choice of the ASCII table
_CHAR_ASCII = _CHAR_ASCII_10


# Get the size of the resized by keeping width / height ratio and with the
# resulting smaller or equal to _MAX_WIDTH x _MAXH_HEIGHT
def getResizedDim(frame):
    rows, cols = frame.shape[:2]

    factorH = 1. * _MAX_HEIGHT / rows
    factorW = 1. * _MAX_WIDTH / cols

    factor = min(factorH, factorW)

    return (int(rows * factor), int(cols * factor))


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


def main():
    # Create a VideoCapture object
    cam = cv2.VideoCapture(0)

    # Grab a frame
    ret, frame = cam.read()
    if not ret:
        return
    # Get the size of the resized images
    sizeSmall = getResizedDim(frame)

    # Resulting 2D array to display in the console
    resultASCII = np.chararray(sizeSmall)

    # Step for ASCII mapping
    stepASCII = 255. / len(_CHAR_ASCII)

    while(True):
        # Grab a frame
        ret, frame = cam.read()
        if not ret:
            break

        # Display the current image
        cv2.imshow("ASCII Art", frame)

        # Convert and resize the grabbed frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resizedGray = cv2.resize(gray, sizeSmall[::-1])

        # Map the ASCII table with the 0->255 grayscal values
        for i, c in enumerate(_CHAR_ASCII):
            resultASCII[np.where(np.logical_and(
                    i * stepASCII <= resizedGray,
                    resizedGray < (i + 1) * stepASCII))] = c

        # Display the ASCII art in the console
        for row in resultASCII:
            print(row.tostring())
        print("\n")

        key = cv2.waitKey(20)
        # ESC - quit the app
        if key == 27:
            break

    # Destroy the windows
    cv2.destroyAllWindows()
    # Release the webcam
    cam.release()

if __name__ == '__main__':
    main()
