import cv2
import numpy as np
import json
from perception.detection import object_detection
from perception.transformation import pixel_to_robot

ROBOT_IP = "192.168.1.6"
def workspace_capture():
    camera = cv2.VideoCapture(2)
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = camera.read()
        cv2.imshow('Camera',frame)
        cv2.waitKey(0)
        answer  = input("is the workspace empty? (y/n):")
        if answer.lower() == 'y':
            cv2.imwrite('empty.jpg', frame)
            print("Image saved as empty.jpg")
            camera.release()
            cv2.destroyAllWindows()
            break
        elif answer.lower() == 'n':
            print("...")
        else:
            print("please enter y or n.")

def calibration():
    from robot.dobot_controller import GetCurrentPosition, StartFeedbackThread, ConnectRobot

    while True:
        answer = input("Ready for calibration ? (y/n):")
        if answer.lower() == 'y':
            print("calibrating")
            break
        elif answer.lower() =='n':
            print("...")
        else:
            print("please enter y or n.")

    robot_pts_list = []
    img_pts = object_detection()

    dashboard, move, feed = ConnectRobot(ip=ROBOT_IP, timeout_s=5.0)

    feed_thread = StartFeedbackThread(feed)

    for i in range(len(img_pts)):
        while True:
            answer = input(f"Is robot in position{i+1}? (y/n):")
            if answer.lower() == 'y':
                pos = GetCurrentPosition()
                robot_pts_list.append([pos[0],pos[1]])
                break
            elif answer.lower() == 'n':
                print("...")
            else:
                print("please enter y or n.")

    robot_pts = np.array(robot_pts_list, dtype=np.float32)
    print(robot_pts)

    H, mask = cv2.findHomography(img_pts, robot_pts, method=0)

    X_pred, Y_pred = pixel_to_robot(456, 252, H)
    print("Predicted:", X_pred, Y_pred)

    print("Homography matrix H:")
    print(H)

    H = H.tolist()
    H = {"H": H}

    with open("calibration.json", "w") as f:
        json.dump(H, f, indent=4)

if __name__ == "__main__":
    workspace_capture()
    calibration()
