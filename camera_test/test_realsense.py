import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# Włącz RGB i Depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Daj kamerze chwilę na złapanie ekspozycji
for i in range(5):
    pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())

# Zapis obrazów
cv2.imwrite("color.jpg", color_image)
cv2.imwrite("depth.png", depth_image)

# Kolorowa mapa głębi
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
cv2.imwrite("depth_colormap.jpg", depth_colormap)

print("✅ Zapisano pliki: color.jpg, depth.png, depth_colormap.jpg")

pipeline.stop()
