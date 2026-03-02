import cv2
import numpy as np

def object_detection():

    camera = cv2.VideoCapture(2)
    #frame = cv2.imread('/home/U2CN/Documents/School/machine-vision/project/fig4.png')
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("maskG", cv2.WINDOW_NORMAL)
    cv2.namedWindow("maskR", cv2.WINDOW_NORMAL)
    cv2.namedWindow("maskB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("maskY", cv2.WINDOW_NORMAL)
    cv2.namedWindow("maskO", cv2.WINDOW_NORMAL)
    cv2.namedWindow("a", cv2.WINDOW_NORMAL)

    ret, frame = camera.read()


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
#-------------------------------------------------------------
# Green
#-------------------------------------------------------------
    lower_green = np.array([60, 30 , 15])
    upper_green = np.array([100, 255, 255])
    maskG = cv2.inRange(hsv, lower_green, upper_green)
    maskG = cv2.erode(maskG, kernel, iterations=5)
    maskG = cv2.dilate(maskG, kernel, iterations=5)
#-------------------------------------------------------------
# Red
#-------------------------------------------------------------
    lower_red1 = np.array([0, 35, 35])
    upper_red1 = np.array([5, 235, 235])
    lower_red2 = np.array([160, 35, 35])
    upper_red2 = np.array([179, 235, 235])
    maskR1 = cv2.inRange(hsv, lower_red1, upper_red1)
    maskR2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskR = np.max([maskR1,maskR2],axis= 0)
    maskR = cv2.erode(maskR, kernel, iterations=5)
    maskR = cv2.dilate(maskR, kernel, iterations=5)
#-------------------------------------------------------------
# Blue
#-------------------------------------------------------------
    lower_Blue = np.array([100, 25, 25])
    upper_Blue = np.array([160, 255, 255])
    maskB = cv2.inRange(hsv, lower_Blue, upper_Blue)
    maskB = cv2.erode(maskB, kernel, iterations=3)
    maskB = cv2.dilate(maskB, kernel, iterations=5)
#-------------------------------------------------------------
# Yellow
#-------------------------------------------------------------
    lower_yellow = np.array([22, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    maskY = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskY = cv2.erode(maskY, kernel, iterations=3)
    maskY = cv2.dilate(maskY, kernel, iterations=5)
#-------------------------------------------------------------
# Orange
#-------------------------------------------------------------
    # lower_orange = np.array([6, 187, 112])
    # upper_orange = np.array([20, 255, 255])
    # maskO = cv2.inRange(hsv, lower_orange, upper_orange)
    # maskO = cv2.dilate(maskO, kernel, iterations=5)
#-------------------------------------------------------------
#
#-------------------------------------------------------------
    mask = np.max([maskG,maskR,maskB,maskY], axis=0)
    mask = cv2.erode(mask, kernel, iterations=6)
#-------------------------------------------------------------



    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    print(f'Found {num_labels-1} objects')

    img_pts_list = []

    for (p1x, p1y, size_x, size_y, _) in stats[1:]:
        print(f'Object inside the rectangle with coordinates ({p1x},{p1y}), ({p1x+size_x}, {p1y+size_y})')
        cv2.rectangle(frame, (p1x,p1y), (p1x+size_x, p1y+size_y), (0,0,255), 2)

    for (cx, cy) in centroids[1:]:
        cx, cy = int(cx), int(cy)
        img_pts_list.append([cx, cy])
        print(f'Centroid: ({cx},{cy})')
        cv2.circle(frame, (cx,cy), 2, (191,40,0), 2)
        cv2.putText(frame, f"({cx},{cy})" , (cx-40, cy-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 125), 1, cv2.LINE_AA)

    img_pts = np.array(img_pts_list, dtype=np.float32)
    print(img_pts)

    cv2.imshow('Original',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('maskG',maskG)
    cv2.imshow('maskR',maskR)
    cv2.imshow('maskB',maskB)
    cv2.imshow('maskY',maskY)
    #cv2.imshow('maskO',maskO)
    cv2.imshow('a',frame)
    cv2.waitKey(0)

    camera.release()
    cv2.destroyAllWindows()

    return img_pts
