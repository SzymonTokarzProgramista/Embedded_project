import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hailo
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection_simple.detection_pipeline_simple import GStreamerDetectionApp
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer

import cv2
import numpy as np


class user_app_callback_class(app_callback_class):
    def __init__(self, output_path="output.mp4", fps=30, width=640, height=480):
        super().__init__()
        self.total_people = 0
        self.total_frames = 0
        self.i = 0
        self.use_frame = True  # Enable frame extraction
        self.video_writer = None
        self.output_path = output_path

def app_callback(pad, info, user_data):
    user_data.increment()
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Convert GStreamer buffer to numpy array (Hailo SDK should provide shape info)
    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)
    if width is None or height is None:
        print("Warning: Missing width/height from caps")
        return Gst.PadProbeReturn.OK

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    
    # Get video frame
    frame = get_numpy_from_buffer(buffer, format, width, height)
    
    if frame is None:
        print("Warning: Empty frame, skipping.")
        return Gst.PadProbeReturn.OK

    print("Processing buffer for detections...")
    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    people_count = 0
    for det in detections:

        label = det.get_label()
        confidence = det.get_confidence()
        if label == "person":
            people_count += 1

    # Update stats
    user_data.total_people += people_count
    user_data.total_frames += 1
    running_average = user_data.total_people / max(user_data.total_frames, 1)

    # Print stats
    print(f"Frame count: {user_data.get_count()}")
    print(f"People detected in frame: {people_count}")
    print(f"Running average people per frame: {running_average:.2f}")

    # Save or push frame to output video
    if user_data.i < 100:
        bbox = det.get_bbox()  # bounding box coordinates     
        h, w, _ = frame.shape   
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)

        print(f"height: {h}, width: {width}, BBox: ({x1}, {y1}), ({x2}, {y2})")
        # Draw rectangle and label
        frame = np.ascontiguousarray(frame).astype(np.uint8).copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if user_data.video_writer is None:            
            h, w, _ = frame.shape
            user_data.video_writer = cv2.VideoWriter(
                user_data.output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (w, h)
        )
        user_data.set_frame(frame)
        user_data.video_writer.write(frame)
        print(f"frame shape: {frame.shape}")
    else:
        user_data.video_writer.release()

    cv2.imwrite(f"debug_frame.jpg", frame)


    user_data.i = user_data.i + 1
    print(user_data.i)
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class(output_path="output.mp4")
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
    user_data.video_writer.release()
