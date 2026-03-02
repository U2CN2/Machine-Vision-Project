import numpy as np
import json
from perception.detection import object_detection

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    pr = H @ p
    pr = pr / pr[2, 0]
    X = pr[0, 0]
    Y = pr[1, 0]
    return X, Y

def robot_coordinates():
    target_points = []
    with open("calibration.json", "r") as f:
        data = json.load(f)
    H = np.array(data["H"], dtype=float)
    img_pts = object_detection()
    for x,y in img_pts:
        X, Y = pixel_to_robot(x,y,H)
        target_points.append([X, Y,-165,0])
    return target_points
