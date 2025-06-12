from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict
from queue import PriorityQueue

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

# Define grid-based environment and obstacles
GRID_WIDTH = 10
GRID_HEIGHT = 10
obstacles = set()

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(grid, start, goal):
    open_set = PriorityQueue()
    open_set.put(start, 0)
    came_from = {}
    g_score = {node: float('inf') for node in grid}
    g_score[start] = 0
    f_score = {node: float('inf') for node in grid}
    f_score[start] = heuristic(start, goal)

    while not open_set.empty():
        current = open_set.get()

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1  # Assuming each step has a cost of 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_set.queue:
                    open_set.put(neighbor, f_score[neighbor])

    return None  # No path found

def neighbors(node):
    # Define possible neighbors of a node
    x, y = node
    return [(x + dx, y + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]

def convert_frame_to_grid(frame):
    # Convert the video frame into a grid representation
    # Define this function based on the size of your environment and obstacles
    # For simplicity, let's assume the grid has the same dimensions as the frame
    grid = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    return grid

def move_robot_to_position(position):
    # Move the robot to the specified position in the grid
    # Implement your robot movement control logic here
    print(f"Moving to position {position}")

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

                    # Update environment based on detected obstacles
                    update_environment([(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))])

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            # Convert the frame to grid
            grid = convert_frame_to_grid(frame)

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

def update_environment(detected_objects):
    # Update environment representation based on detected objects
    global obstacles
    for obj in detected_objects:
        obstacles.add(obj)  # Assuming (x, y) coordinates of detected obstacles

    # Optionally, remove expired obstacles from the environment

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(object_detection_and_tracking(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
