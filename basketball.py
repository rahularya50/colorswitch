# coding=utf-8
import statistics
import time
import math

import cv2
import imutils
import numpy as np
import subprocess

MIN_HISTORY = 3
DELTA_SMALL = math.radians(30)
DELTA_LARGE = math.radians(50)


def selectRectangle(event, x, y, flags, bounds):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    bounds.append((x, y))
    print(x, y)


def clickHandler(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(x, y)


def printCoords(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(x, y)


def initalize(bounds, getColor):
    cv2.namedWindow("raw")
    cv2.namedWindow("scaled")
    cv2.namedWindow("darks")
    cv2.namedWindow("processed")
    cv2.namedWindow("model")
    cv2.setMouseCallback("raw", selectRectangle, param=bounds)
    cv2.setMouseCallback("scaled", getColor)
    cv2.setMouseCallback("processed", clickHandler)

    vc = cv2.VideoCapture(0)
    return vc


def getTrackBoundaries(cnts, width, height):
    a = max(cnts, key=lambda x: cv2.moments(x)["m00"]
    if cv2.moments(x)["m00"] and
       cv2.moments(x)["m00"] / (width * height) < 0.9 and
       height / 2 > cv2.moments(x)['m01'] / cv2.moments(x)["m00"] > 20 else 0)
    b = max(cnts, key=lambda x: cv2.moments(x)["m00"]
    if cv2.moments(x)["m00"] and
       cv2.moments(x)["m00"] / cv2.moments(a)["m00"] < 0.8 and
       cv2.moments(x)['m01'] / cv2.moments(x)["m00"] > height / 2 else 0)
    return a, b


def contour_coords(c):
    x, y, w, h = cv2.boundingRect(c)
    return int(x + w // 2), int(y + h // 2)


def getIntersections(root, model, velocity, track_boundaries, width, height):
    blank = np.zeros((width, height, 3), np.uint8)

    cv2.line(blank, root[:2],
             tuple(i + j for i, j in zip(root, velocity))[:2], (255, 255, 255))

    cv2.line(model, root[:2],
             tuple(i + j for i, j in zip(root, velocity))[:2], (255, 255, 255))

    boundaries = (cv2.drawContours(np.zeros((width, height, 3), np.uint8), track_boundaries, i, (255, 255, 255))
                  for i in range(2))

    intersections = (np.logical_and(blank, s) for s in boundaries)
    point_lists = (np.transpose(np.nonzero(intersection)) for intersection in intersections)

    out_sets = []
    for points in point_lists:
        out_sets.append(set())
        for point in points:
            out_sets[-1].add(tuple(point[:2]))

    for s in out_sets:
        for x, y in s:
            cv2.circle(model, (y, x), 6, (255, 255, 255), -1)

    return out_sets


def getAngle(particle, prev_pos):
    approx_vel = [0, 0]

    if len(prev_pos) > MIN_HISTORY:
        approx_vel = [(i - j) for i, j in zip(contour_coords(particle), prev_pos[-MIN_HISTORY])]
    if approx_vel == [0, 0]:
        approx_vel = [0, -1]

    prev_pos.append(contour_coords(particle))
    (x, y), (w, h), angle = cv2.minAreaRect(particle)

    angle = math.radians(angle)
    if w < h:
        angle += math.pi / 2

    scaled_delta = [math.cos(angle), math.sin(angle)]
    if scaled_delta[0] * approx_vel[0] + scaled_delta[1] * approx_vel[1] < 0:
        angle += math.pi

    return angle


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

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.Canny(blurred, 15, 80)

    thresh_cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    thresh_cnts = thresh_cnts[0] if imutils.is_cv2() else thresh_cnts[1]

    darks = cv2.inRange(cv2.cvtColor(result, cv2.COLOR_BGR2HSV), (0, 100, 100), (50, 255, 255))

    darks_cnts = cv2.findContours(darks.copy(), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    darks_cnts = darks_cnts[0] if imutils.is_cv2() else darks_cnts[1]

    return result, thresh, darks, thresh_cnts, darks_cnts, width, height


def main():
    bounds = []
    prev_pos = []
    shot_count = 0
    prev_shot = 0

    def getColor(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)[y, x])

    vc = initalize(bounds, getColor)

    while True:
        try:
            if cv2.waitKey(10) != 255:
                while True:
                    pass
            raw = getRaw(vc)

            if not bounds:
                cv2.imshow("raw", raw)
                continue
            drawBounds(raw, bounds)
            cv2.imshow("raw", raw)

            if len(bounds) < 4:
                continue
            scaled, thresh, darks, thresh_cnts, darks_cnts, width, height = getDisplay(raw, bounds)
            cv2.imshow("scaled", scaled)
            cv2.imshow("darks", darks)
            cv2.imshow("processed", thresh)

            track_boundaries = getTrackBoundaries(darks_cnts, width, height)

            coords = []
            for c in track_boundaries:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"]) if M["m00"] else 0
                cY = int(M["m01"] / M["m00"]) if M["m00"] else 0
                coords.append((cX, cY))

            coords.sort(key=lambda x: x[1])

            hoop = coords[0]
            bball = coords[1]

            # print("hoop:", hoop, "ball:", bball)

            start_x = bball[0] / height * 1440
            start_y = bball[1] / width * 2880
            # print("{} = {} / {} * 1440".format(start_x, bball[0], height))

            end_x = hoop[0] / height * 1440
            end_y = hoop[1] / width * 2880

            prev_pos.append((end_x, end_y))

            if len(prev_pos) > 5:
                prev_pos = prev_pos[1:]

            # print(end_x, end_y)

            xdelta = prev_pos[-1][0] - prev_pos[-3][0] if len(prev_pos) > 2 else 0

            X_MIN = 189
            X_MAX = 1250

            xdirection = 0
            if shot_count >= 10 and xdelta > 0:
                xdirection = 1
            elif shot_count >= 10:
                xdirection = -1

            if 10 <= shot_count < 20:
                end_x += xdirection * 6 * 80
            elif 20 <= shot_count:
                end_x += xdirection * 10.5 * 80

            xoverflow = True
            if end_x < X_MIN:
                end_x = X_MIN + (X_MIN - end_x)
            elif end_x > X_MAX:
                end_x = X_MAX - (end_x - X_MAX)
            else:
                xoverflow = False

            ydelta = prev_pos[-1][1] - prev_pos[-3][1] if len(prev_pos) > 2 else 0

            Y_MIN = 1180
            Y_MAX = 1404

            ydirection = 0
            if shot_count >= 30 and ydelta > 0:
                ydirection = 1
            elif shot_count >= 30:
                ydirection = -1

            if 30 <= shot_count < 40:
                end_y += ydirection * 2 * 80
            elif shot_count >= 40:
                end_y += ydirection * 5 * 80

            yoverflow = True
            if end_y < Y_MIN:
                end_y = Y_MIN + (Y_MIN - end_y)
            elif end_y > Y_MAX:
                end_y = Y_MAX - (end_y - Y_MAX)
            else:
                yoverflow = False

            model = np.zeros((width, height, 3), np.uint8)
            cv2.drawContours(model, track_boundaries, -1, (0, 255, 0), 2)

            cv2.circle(model, hoop, 7, (255, 255, 0), -1)
            cv2.circle(model, (int(end_x * height / 1440), int(end_y * width / 2880)), 7, (255, 255, 0), -1)
            cv2.circle(model, bball, 7, (255, 0, 255), -1)

            cv2.imshow("model", model)

            # if shot_count == 40:
            #      return

            if time.time() - prev_shot > 3 and (shot_count < 10 or xdelta != 0):
                    # (shot_count < 10 or abs(end_x - start_x) < 400):
                print(xdelta)
                proc = subprocess.Popen('C:\\Android\\platform-tools\\adb shell', shell=True, stdin=subprocess.PIPE)
                proc.communicate(b'input swipe %a %a %a %a' % (start_x, start_y, end_x, end_y))
                prev_shot = time.time()
                print('shot', "zeroed" if xdelta == 0 else "nonzeroed")
                shot_count += 1
        except ZeroDivisionError as e:
            prev_hoop = None
            print("ERR", e)


if __name__ == '__main__':
    main()
