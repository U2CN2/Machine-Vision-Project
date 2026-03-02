import cv2

# 0 is usually the default built-in camera, 1 or 2 for USB cams
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(2)

print("Press 's' to save the image and 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save the image
        cv2.imwrite('captured_image.jpg', frame)
        print("Image saved as captured_image.jpg")
        break
    elif key == ord('q'):
        # Quit without saving
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
