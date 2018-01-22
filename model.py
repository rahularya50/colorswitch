# coding=utf-8

import math
import time
from msvcrt import getch

import cv2
import numpy as np

WHITE = (255, 255, 255)

VEL = 3
ROT_FAST = 6
ROT_SLOW = 4


def printCoords(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(x, y)


def initialize_display():
    cv2.namedWindow("working")
    cv2.setMouseCallback("working", printCoords)

    img = cv2.imread("baseline.png")
    img = cv2.resize(img, (0, 0), img, 0.25, 0.25)

    return img


def render_scene(img, player_coords,
                 x_center=181,
                 y_center=302,
                 exterior_width=272,
                 exterior_height=249,
                 exterior_rad=120,
                 interior_width=98,
                 interior_height=190,
                 interior_rad=40):
    img = np.zeros(img.shape, np.uint8)

    # Draw outer border
    cv2.ellipse(img, (x_center, y_center - exterior_height // 2), (exterior_width // 2, exterior_rad), 180, 0, 180, WHITE)
    cv2.ellipse(img, (x_center, y_center + exterior_height // 2), (exterior_width // 2, exterior_rad), 0, 0, 180, WHITE)
    cv2.line(img, (x_center - exterior_width // 2, y_center - exterior_height // 2),
             (x_center - exterior_width // 2, y_center + exterior_height // 2), WHITE)
    cv2.line(img, (x_center + exterior_width // 2, y_center - exterior_height // 2),
             (x_center + exterior_width // 2, y_center + exterior_height // 2), WHITE)

    # Draw inner border
    cv2.ellipse(img, (x_center, y_center - interior_height // 2), (interior_width // 2, interior_rad), 180, 0, 180, WHITE)
    cv2.ellipse(img, (x_center, y_center + interior_height // 2), (interior_width // 2, interior_rad), 0, 0, 180, WHITE)
    cv2.line(img, (x_center - interior_width // 2, y_center - interior_height // 2),
             (x_center - interior_width // 2, y_center + interior_height // 2), WHITE)
    cv2.line(img, (x_center + interior_width // 2, y_center - interior_height // 2),
             (x_center + interior_width // 2, y_center + interior_height // 2), WHITE)

    # Draw player
    cv2.circle(img, player_coords[:2], 10, WHITE)

    cv2.imshow("working", img)


def update(x, y, vel_angle, orient_angle):
    x += int(math.cos(math.radians(vel_angle)) * VEL)
    y += int(math.sin(math.radians(vel_angle)) * VEL)
    vel_angle += (orient_angle > vel_angle) * min(ROT_SLOW, abs(orient_angle - vel_angle))
    return x, y, vel_angle, orient_angle


def main():
    player_coords = (91, 422, -90, -90)
    img = initialize_display()

    while True:
        player_coords = update(*player_coords)
        render_scene(img, player_coords)
        if cv2.waitKey(15) != 255:
            player_coords = (*player_coords[:3], player_coords[3] + ROT_FAST)


if __name__ == '__main__':
    main()
