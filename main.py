import cv2
import os
from ultralytics import YOLO
from pyzbar.pyzbar import decode
import numpy as np

# Load the YOLOv8 model
model = YOLO("qrcode.pt")

# Define the directory to save cropped images
crop_dir = "crop"
os.makedirs(crop_dir, exist_ok=True)

# Open the video file or camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Get the detections
        detections = results[0].boxes

        # Loop through detections and process them
        for i, detection in enumerate(detections):
            # Get bounding box coordinates and class id
            box = detection.xyxy[0].cpu().numpy().astype(int)  # Convert to int
            class_id = detection.cls[0].cpu().numpy().astype(int)  # Convert to int

            # Crop the detected object from the frame
            x1, y1, x2, y2 = box
            cropped_image = frame[y1:y2, x1:x2]

            # Save the cropped image
            crop_path = os.path.join(crop_dir, f"crop_{i}.png")
            cv2.imwrite(crop_path, cropped_image)

            # Preprocess the cropped image (if needed)
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            decoded_objects = decode(processed_image)
            for obj in decoded_objects:
                print("Detected QR Code:", obj.data.decode('utf-8'))
            # Nhận diện mã vạch hoặc QR code từ hình ảnh gốc (không phải từ hình ảnh đã qua tiền xử lý)
            for barcode in decode(cropped_image):
                myData = barcode.data.decode('utf-8')
                print(myData)
                pts = np.array([barcode.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(cropped_image, [pts], True, (255, 0, 255), 5)
                pts2 = barcode.rect
                cv2.putText(cropped_image, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()