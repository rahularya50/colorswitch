# coding=utf-8

import time
import math

import cv2
import imutils
import numpy as np

import serial

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
    if cv2.moments(x)["m00"] / (width * height) < 0.9 else 0)
    b = max(cnts, key=lambda x: cv2.moments(x)["m00"]
    if cv2.moments(x)["m00"] / cv2.moments(a)["m00"] < 0.8 else 0)
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


def getParticle(model, cnts, prev_pos):
    particle = min(cnts, key=lambda x: sum(dist(a, contour_coords(x)) for a in prev_pos[-MIN_HISTORY:]))
    cv2.drawContours(model, [particle], -1, (255, 255, 255))
    return particle


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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.Canny(blurred, 50, 130)

    thresh_cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    thresh_cnts = thresh_cnts[0] if imutils.is_cv2() else thresh_cnts[1]

    darks = cv2.inRange(cv2.cvtColor(result, cv2.COLOR_BGR2HSV), (0, 0, 0), (180, 255, 80))

    darks_cnts = cv2.findContours(darks.copy(), cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)
    darks_cnts = darks_cnts[0] if imutils.is_cv2() else darks_cnts[1]

    return result, thresh, darks, thresh_cnts, darks_cnts, width, height


class Arduino:
    def __init__(self):
        self.ser = serial.Serial('COM3', 19200)
        time.sleep(2)
        self.pressed = False

    def press(self):
        if not self.pressed:
            self.ser.write(bytes(" ", "utf-8"))
            self.pressed = True

    def release(self):
        if self.pressed:
            self.ser.write(bytes(" ", "utf-8"))
            self.pressed = False

    def tap(self):
        if self.pressed:
            self.release()
        self.ser.write(bytes("  ", "utf-8"))


def main():
    track_boundaries = None
    bounds = []
    prev_pos = []

    turnAngle = 0
    releaseY = 0

    controller = Arduino()

    def getColor(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)[y, x])

    vc = initalize(bounds, prev_pos, getColor)

    while True:
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

        # width exceeds height (they're the other way around!)

        if not prev_pos:
            track_boundaries = getTrackBoundaries(thresh_cnts, width, height)

        model = np.zeros((width, height, 3), np.uint8)
        cv2.drawContours(model, track_boundaries, -1, (0, 255, 0), 2)

        if not prev_pos:
            cv2.imshow("model", model)
            continue

        particle = getParticle(model, darks_cnts, prev_pos)
        angle = getAngle(particle, prev_pos)

        x, y = contour_coords(particle)

        rot_angle = math.atan2(y - height // 2, x - width // 2)

        velocity = getDirectionVector(angle + math.pi / 6 - (angle + math.pi / 6) % (math.pi / 2))
        # radial_in = getDirectionVector(angle + math.pi / 2)
        # radial_out = getDirectionVector(angle - math.pi / 2)

        # velocity = getDirectionVector((math.pi * 1 / 2 if angle % (math.pi * 2) < math.pi else (math.pi * 3 / 2)))
        radial_in = getDirectionVector((math.pi * 1 / 2 if angle % (math.pi * 2) < math.pi else (math.pi * 3 / 2)) + math.pi / 2)
        radial_out = getDirectionVector((math.pi * 1 / 2 if angle % (math.pi * 2) < math.pi else (math.pi * 3 / 2)) - math.pi / 2)

        # offset_small = getDirectionVector(angle + DELTA_SMALL)
        # offset_large = getDirectionVector(angle + DELTA_LARGE)

        intersect_1, intersect_2 = getIntersections((x, y), model, velocity, track_boundaries, width, height)

        # projected_pos = (x + velocity[0], y + velocity[1])

        # intersect_3, intersect_4 = getIntersections(particle, model, offset_small, track_boundaries, width, height)
        # intersect_5, intersect_6 = getIntersections(particle, model, offset_large, track_boundaries, width, height)
        # intersect_7, intersect_8 = getIntersections((x, y), model, radial_in, track_boundaries, width, height)
        # intersect_9, intersect_10 = getIntersections((x, y), model, radial_out, track_boundaries, width, height)

        # if not intersect_6:
        #     controller.press()
        # elif intersect_1 and dist(contour_coords(particle), reversed(next(iter(intersect_1)))) < 20000:
        #     controller.tap()
        # elif intersect_4:
        #     controller.release()

        # elif ((angle % math.pi) > math.pi / 3):

        if intersect_1:
            d = dist((x, y), reversed(next(iter(intersect_1))))

            if 37 < x < 112:
                x_dev = (x - 75) / 37
            elif 190 < x < 260:
                x_dev = (x - 225) / 35
            else:
                x_dev = None

            MAX_ANG_DEV = 0.1637  # radians
            ang_dev = MAX_ANG_DEV * x_dev if x_dev else None
            if d < 15000:  # or (y < 164 and x < 151) or (y > 341 and x > 151):
                controller.press()
                releaseY = y
            elif (angle % (math.pi / 2)) > math.pi * (1/2 - 1/6) or (angle % (math.pi / 2)) < math.pi * (1/8):
                print("release!")
                controller.release()

        # if intersect_1:
        #     d = dist((x, y), reversed(next(iter(intersect_1))))
        #     if d < 15000:
        #         pass
        #         # controller.press()
        #
        # if intersect_8:
        #     closest_in = min(dist(a, reversed(projected_pos)) for a in intersect_8)
        #     closest_out = min(dist(a, reversed(projected_pos)) for a in intersect_9) if intersect_9 else float("inf")
        #
        #     print(closest_out, closest_in)
        #
        #     if closest_in / closest_out > 1.2 and (angle % math.pi) > math.pi / 3:
        #         controller.press()
        #     else:
        #         controller.release()


        cv2.imshow("model", model)


if __name__ == '__main__':
    main()
