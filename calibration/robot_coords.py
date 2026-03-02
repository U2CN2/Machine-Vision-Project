from robot.dobot_api import DobotApiDashboard
import time


dash = DobotApiDashboard("192.168.1.6", 29999)

def parse_pose(reply):
    start = reply.find("{")
    end = reply.find("}")
    return [float(x) for x in reply[start+1:end].split(",")]

pose_reply = dash.GetPose()
x, y, z, rx, ry, rz = parse_pose(pose_reply)
print(f"Current pose: X={x}, Y={y}, Z={z}")
