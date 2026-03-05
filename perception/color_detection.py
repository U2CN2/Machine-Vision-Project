
import cv2
import numpy as np

def detect_and_label(frame, mask, color_name, box_color):
    # Clean mask: remove small noise and fill gaps in the tiles
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Filter out noise (adjust if tiles are smaller/larger)
        if area < 500:
            continue

        cx, cy = int(centroids[i][0]), int(centroids[i][1])

        # Draw the bounding box and center point
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # --- Labeling with Coordinates  ---
        label = f"{color_name} ({cx}, {cy})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        
        # Position text: 10px above box. If too high, put it below the box.
        text_y = y - 10 if y - 20 > 0 else y + h + 20
        
        # Draw background box for text readability
        cv2.rectangle(frame, (x, text_y - th - 4), (x + tw + 2, text_y + 4), box_color, -1)
        
        # Text color: Black for light boxes, White for dark boxes
        t_color = (0, 0, 0) if color_name in ["YELLOW", "ORANGE"] else (255, 255, 255)
        
        cv2.putText(frame, label, (x + 2, text_y), font, scale, t_color, thickness, cv2.LINE_AA)

def object_detection():
    # Load your image
    path = "/Users/kabs/Downloads/class/robotics/opencv/captured_image.jpg"
    frame = cv2.imread(path)

    if frame is None:
        print("Image file not found!")
        return

    # Blur reduces glare interference
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # ---------------- COLOR DEFINITIONS ----------------
    
    # RED (Handles two ranges at the edge of the hue scale)
    maskR = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 150, 60]), np.array([6, 255, 255])),
        cv2.inRange(hsv, np.array([170, 150, 60]), np.array([180, 255, 255]))
    )

    # ORANGE
    maskO = cv2.inRange(hsv, np.array([7, 150, 100]), np.array([18, 255, 255]))

    # YELLOW
    maskY = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

    # GREEN (Bright)
    maskG = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([85, 255, 255]))

    # DEEP GREEN (Low brightness/Value for dark tiles)
    maskDG = cv2.inRange(hsv, np.array([40, 40, 20]), np.array([90, 255, 90]))

    # BLUE
    maskB = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))

    # PURPLE (Adjusted for dark purple tiles)
    maskP = cv2.inRange(hsv, np.array([135, 40, 20]), np.array([165, 255, 150]))

    # ---------------- RUN DETECTION ----------------

    detect_and_label(frame, maskR, "RED", (0, 0, 255))
    detect_and_label(frame, maskO, "ORANGE", (0, 165, 255))
    detect_and_label(frame, maskY, "YELLOW", (0, 255, 255))
    detect_and_label(frame, maskG, "GREEN", (0, 255, 0))
    detect_and_label(frame, maskDG, "D-GREEN", (0, 100, 0))
    detect_and_label(frame, maskB, "BLUE", (255, 0, 0))
    detect_and_label(frame, maskP, "PURPLE", (128, 0, 128))

    cv2.imshow("Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection()




