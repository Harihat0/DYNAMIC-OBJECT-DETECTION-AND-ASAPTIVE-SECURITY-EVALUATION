from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

app = Flask(__name__)

# Initialize the object detection model
model = YOLO("yolov3-tinyu.pt")
names = model.model.names

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to the video file

if not cap.isOpened():
    print("Error: Failed to open video capture")
else:
    print("Video capture opened successfully")

# Initialize track history
track_history = defaultdict(lambda: [])

# Movement commands
MOVEMENT_MAP = {
    ord('w'): "Forward",
    ord('a'): "Left",
    ord('s'): "Backward",
    ord('d'): "Right",
    ord('q'): "Stop"
}

def object_detection_and_tracking():
    while cap.isOpened():
        # Read a frame from the video capture
        success, frame = cap.read()
        if success:
            # Perform object detection on the frame
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:
                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Annotate the detected objects on the frame
                annotator = Annotator(frame, line_width=2)
                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Check for keyboard input for movement control
            key = cv2.waitKey(1)
            if key in MOVEMENT_MAP:
                movement = MOVEMENT_MAP[key]
                print("Movement command:", movement)  # Replace this with your actual robot control logic

            # Break the loop if 'q' is pressed
            if key & 0xFF == ord("q"):
                break
        else:
            print("Error: Failed to read frame from video capture")
            break

@app.route('/')
def index():
    return render_template('index.html')

# Define your movement control logic here
def move_robot(direction):
    
    print(f"Moving {direction}")

@app.route('/move', methods=['POST'])
def move():
    # Get the movement direction from the request
    direction = request.json['direction']
    
    # Call the function to move the robot
    move_robot(direction)
    
    # Return a response
    return 'OK', 200

@app.route('/video_feed')
def video_feed():
    return Response(object_detection_and_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
