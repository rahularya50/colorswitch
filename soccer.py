# coding=utf-8

import math
import subprocess

import cv2
import imutils
import numpy as np

import interaction

MIN_HISTORY = 3
DELTA_SMALL = math.radians(30)
DELTA_LARGE = math.radians(50)


def selectRectangle(event, x, y, flags, bounds):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    bounds.append((x, y))
    print(x, y)


def clickHandler(event, x, y, flags, prev_pos):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    prev_pos.clear()
    prev_pos.append((x, y))
    print(x, y)


def printCoords(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(x, y)


def initalize(bounds, prev_pos, getColor):
    cv2.namedWindow("raw")
    cv2.namedWindow("scaled")
    cv2.namedWindow("darks")
    cv2.namedWindow("processed")
    cv2.namedWindow("model")
    cv2.setMouseCallback("raw", selectRectangle, param=bounds)
    cv2.setMouseCallback("scaled", getColor)
    cv2.setMouseCallback("processed", clickHandler, param=prev_pos)

    vc = cv2.VideoCapture(0)
    return vc


def getTrackBoundaries(cnts, width, height):
    a = max(cnts, key=lambda x: cv2.moments(x)["m00"]
    if cv2.moments(x)["m00"] and
       cv2.moments(x)["m00"] / (width * height) < 0.9 and
       cv2.moments(x)['m01'] / cv2.moments(x)["m00"] > 20 else 0)
    return a


def contour_coords(c):
    x, y, w, h = cv2.boundingRect(c)
    return int(x + w // 2), int(y + h // 2)


def getDirectionVector(angle, scale_factor=10000):
    scaled_delta = [int(scale_factor * math.cos(angle)), int(scale_factor * math.sin(angle))]
    return scaled_delta


def dist(a, b):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(a, b))


def drawBounds(raw, bounds):
    temp = bounds[0]
    for i in bounds:
        cv2.line(raw, temp, i, (0, 255, 0), 5)
        temp = i


def getRaw(vc):
    frame = vc.read()[1]
    return frame


def getDisplay(frame, bounds):
    a, b, c, d = bounds

    widthA = np.sqrt((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2)
    widthB = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    heightA = np.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)
    heightB = np.sqrt((a[0] - d[0]) ** 2 + (a[1] - d[1]) ** 2)

    width = max(int(widthA), int(widthB))
    height = max(int(heightA), int(heightB))

    rect = np.zeros((4, 2), dtype="float32")
    rect[0], rect[1], rect[2], rect[3] = a, b, c, d
    dest = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dest)
    result = cv2.warpPerspective(frame, M, (width, height))
    center = (height / 2, width / 2)
    rotate = cv2.getRotationMatrix2D(center, 90, 1)
    rot_move = np.dot(rotate, np.array([(height - width) / 2, (width - height) / 2, 0]))
    rotate[0, 2] += rot_move[0]
    rotate[1, 2] += rot_move[1]
    result = cv2.warpAffine(result, rotate, (height, width))

    darks = cv2.inRange(cv2.cvtColor(result, cv2.COLOR_BGR2HSV), (0, 0, 0), (180, 255, 80))

    darks_cnts = cv2.findContours(darks, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    darks_cnts = darks_cnts[0] if imutils.is_cv2() else darks_cnts[1]

    return result, darks, darks_cnts, width, height


def main():
    bounds = []
    prev_pos = []
    prev_shot = 0
    prev_hoop = None
    moving = False

    def getColor(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)[y, x])

    vc = initalize(bounds, prev_pos, getColor)

    while True:
        try:
            k = cv2.waitKey(10)
            raw = getRaw(vc)

            if not bounds:
                cv2.imshow("raw", raw)
                continue
            drawBounds(raw, bounds)
            cv2.imshow("raw", raw)

            if len(bounds) < 4:
                continue
            scaled, darks, darks_cnts, width, height = getDisplay(raw, bounds)
            # cv2.imshow("scaled", scaled)
            # cv2.imshow("darks", darks)
            # cv2.imshow("processed", thresh)

            c = getTrackBoundaries(darks_cnts, width, height)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
            cY = int(M["m01"] / M["m00"]) if M["m00"] else 0

            # print(cY)

            start_x = cX / height * 1440
            start_y = cY / width * 2880
            # print("{} = {} / {} * 1440".format(start_x, bball[0], height))

            model = np.zeros((width, height, 3), np.uint8)
            cv2.drawContours(model, [c], -1, (0, 255, 0), 2)

            cv2.circle(model, (cX, cY), 7, (255, 0, 255), -1)

            cv2.imshow("model", model)

            if k == ord(' '):
                proc = subprocess.Popen('C:\\Android\\platform-tools\\adb shell', shell=True, stdin=subprocess.PIPE)
                proc.stdin.wr(b'input tap %a %a\n' % (start_x, start_y))
                # interaction.tap(start_x, start_y)
                print('shot')

        except ZeroDivisionError as e:
            print(e)
            prev_hoop = None


if __name__ == '__main__':
    main()
