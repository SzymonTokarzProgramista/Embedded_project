# Python example using mecanum_pb2.py
import mecanum_pb2
import serial
import time
import math

# Open serial connection
ser = serial.Serial('/dev/ttyUSB0', 19200, timeout=1)  # Adjust port as needed
time.sleep(2)

print('Open serial')

try:
    while True:
        request = mecanum_pb2.ControlRequest()
        request.speed_mmps = 50    # Move forward at 0-500 mm/s
        request.rad = 0.5 * math.pi     # Straight line 0-2*PI, pi/2 is straight, 0 is right
        request.omega = -0.0         # Rotation speed in rad/s, positive is CCW, range 0-2 rad/s
        
        serialized_data = request.SerializeToString()
        ser.write(serialized_data)
        time.sleep(0.1)  # Send every 100ms
        
except KeyboardInterrupt:
    # Stop the robot before exiting
    request.speed_mmps = 0
    request.rad = 0.0          # Straight line
    request.omega = 0.0        # No rotation
    ser.write(request.SerializeToString())
    time.sleep(2)
    ser.close()
