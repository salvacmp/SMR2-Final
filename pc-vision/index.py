import tkinter as tk
from tkinter import Label, Frame, Entry, Button, ttk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import threading
import time
import random
import socket
import datetime
import struct
import pyrealsense2 as rs
import numpy as np
import math
import json

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control")

        # Create main layout with 2 columns
        self.left_frame = Frame(root, width=500, height=500, bg="black")
        self.left_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        self.right_top_frame = Frame(root, width=500, height=250, bg="gray")
        self.right_top_frame.grid(row=0, column=1, padx=10, pady=10)

        self.right_bottom_frame = Frame(root, width=500, height=250, bg="white")
        self.right_bottom_frame.grid(row=1, column=1, padx=10, pady=10)

         # Setting up styles for buttons
        self.style = ttk.Style()
      
        # Use the 'alt' theme for a more modern look
        self.style.theme_use('alt')  

        # Style for buttons
        self.style.configure("TButton",
                             padding=6,
                             relief="flat",  # Flat button for a more modern effect
                             background="#4CAF50",  # Nice green background color
                             foreground="white",  # White text
                             font=("Arial", 12, "bold"))

        # Hover effects for buttons
        self.style.map("TButton",
                       background=[("active", "#45a049")])  # Lighter green color when hovering
        
        # New style for the "Stop" button (red background)
        self.style.configure("RedButton.TButton",
                             padding=6,
                             relief="flat",  # Flat button without 3D effect
                             background="#FF5733",  # Red for the Stop button
                             foreground="white",  # White text
                             font=("Arial", 12, "bold"))
       
        # Hover effect for the "Stop" button
        self.style.map("RedButton.TButton",
                      background=[("active", "#FF2A00")])  # Darker red when hovering
        
         # New style for the "Load" button (red background)
        self.style.configure("BlueButton.TButton",
                             padding=6,
                             relief="flat",  # Flat button without 3D effect
                             background="#4aa2ef",  # Blue for Load button
                             foreground="white",  # White text
                             font=("Arial", 12, "bold"))
       
        # Hover effect for the "Stop" button
        self.style.map("BlueButton.TButton",
                      background=[("active", "#4b73d3")])  # Darker Blue when hovering

        # Ensure that all columns have the same 'weight' so they scale evenly
        for col in range(4):
            self.right_top_frame.grid_columnconfigure(col, weight=1)

        self.right_top_frame.grid_rowconfigure(0, weight=1)
        self.right_top_frame.grid_rowconfigure(1, weight=1)
        self.right_top_frame.grid_rowconfigure(2, weight=1)
        self.right_top_frame.grid_rowconfigure(3, weight=1)
        self.right_top_frame.grid_rowconfigure(4, weight=1)
        self.logo_image = ImageTk.PhotoImage(file="rpx-smr.png")
        self.logo_label = Label(root, image=self.logo_image, bg="white")
        self.logo_label.place(relx=0.5, y=30, anchor="center")

        # Live camera feed setup
        self.camera_label = Label(self.left_frame)
        self.camera_label.pack()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            pass

        #Detection config
        self.model_path = "models/bestn3.pt"
        self.model = YOLO(self.model_path)
        
        #Points Memory
        self.points = []
        self.PointRequest = False
        self.PrecissionRequest = False
        self.PrecissionId = 0
        
        self.ip_label = Label(self.right_top_frame, text="IP:", font=("Arial", 12), bg="gray", fg="white")
        self.ip_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.ip_entry = Entry(self.right_top_frame, width=20)
        self.ip_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.port_label = Label(self.right_top_frame, text="Port:", font=("Arial", 12), bg="gray", fg="white")
        self.port_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.port_entry = Entry(self.right_top_frame, width=20)
        self.port_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Load Button
        self.load_button = ttk.Button(self.right_top_frame, text="Load", command=self.load, style="BlueButton.TButton")
        self.load_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")


        # Buttons for Connect and Disconnect
        self.connect_button = ttk.Button(self.right_top_frame, text="Connect", command=self.connect)
        self.connect_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.disconnect_button = ttk.Button(self.right_top_frame, text="Disconnect", command=self.disconnect)
        self.disconnect_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # Buttons for Start, Pause
        self.start_button = ttk.Button(self.right_top_frame, text="Start", command=self.start_process, state="disabled")
        self.start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.pause_button = ttk.Button(self.right_top_frame, text="Pause", command=self.pause_process, state="disabled")
        self.pause_button.grid(row=3, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # Stop button (spanning 2 columns)
        self.stop_button = ttk.Button(self.right_top_frame, text="Stop", command=self.stop_process, state="disabled", style="RedButton.TButton")
        self.stop_button.grid(row=4, column=0, columnspan=4, padx=5, pady=5, sticky="ew")

        # Logs section
        self.log_listbox = tk.Listbox(self.right_bottom_frame, height=15, width=150)
        self.log_listbox.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        self.log_scrollbar = tk.Scrollbar(self.right_bottom_frame)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_listbox.config(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.config(command=self.log_listbox.yview)

        # Load IP and Port from JSON
        self.load_config()

        # Socket server setup
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 2025))
        self.server_socket.listen(5)
        
        # Robot Socket
        self.robot_socket = None
        
        #tracker
        self.precission_roi_x = 345
        self.precission_roi_y = 285
        self.factor = 20
        self.precissionPoint = [0, 0, 0, 0, 0]
        self.precissionPolygon = []
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recording = cv2.VideoWriter( 
        f"output-{time.time()}.avi", self.fourcc, 60, (640, 480))
        # Start threads for camera and log updates
        self.running = True
        self.active = False
        threading.Thread(target=self.update_camera_feed, daemon=True).start()
        threading.Thread(target=self.socket_server, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    def load_config(self):
        try:
            with open("config.json", "r") as file:
                config = json.load(file)
                self.ip_entry.insert(0, config.get("ip", ""))
                self.port_entry.insert(0, config.get("port", ""))
        except FileNotFoundError:
            with open("config.json", "w") as file:
                json.dump({"ip": "", "port": ""}, file)

    def save_config(self):
        config = {
            "ip": self.ip_entry.get(),
            "port": self.port_entry.get()
        }
        with open("config.json", "w") as file:
            json.dump(config, file)

    def connect(self):
        self.save_config()
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            
            self.robot_socket.connect((self.ip_entry.get(), int(self.port_entry.get())))
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            self.insertLog("Connected to IP: {} Port: {}".format(self.ip_entry.get(), self.port_entry.get()))
            self.insertLog(f"{str(data)}")

            if data == "Connected: Universal Robots Dashboard Server":
                self.insertLog(f"{data}")
                self.active = True
                self.update_button_states()
            else:
                self.insertLog("Failed to connect to the robot.")
                self.robot_socket.close()
                self.active = False
                self.update_button_states()
        except Exception as e:
            self.insertLog(f"Error: {e}")
            self.active = False
            self.update_button_states()
    def load(self):
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self.active:
                self.robot_socket.connect((self.ip_entry.get(), int(self.port_entry.get())))
                load_program = "load /programs/ropax/main.urp\n"
                self.robot_socket.send(load_program.encode())
                data = self.robot_socket.recv(1024)
                data = data.decode('utf-8').strip()
                self.insertLog(f"Loaded: Program")
                self.active = True
                self.update_button_states()
                # self.robot_socket.close()
            else:
                self.insertLog("Failed to connect to the robot.")
                self.robot_socket.close()
                self.update_button_states()
        except Exception as e:
            self.insertLog(f"Error: {e}")
            self.update_button_states()

    def disconnect(self):
        self.robot_socket.close()
        self.insertLog("Disconnected from IP: {} Port: {}".format(self.ip_entry.get(), self.port_entry.get()))
        self.active = False
        self.update_button_states()
        self.recording.release()

    def update_button_states(self):
        state = "normal" if self.active else "disabled"
        self.start_button.config(state=state)
        self.pause_button.config(state=state)
        self.stop_button.config(state=state)
    
    def start_process(self):
        try:
           
            start_program = "play\n"
            self.robot_socket.sendall(start_program.encode())
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            
            self.insertLog(f"{data}")
            self.insertLog("Process started.")
        except Exception as e:
            self.insertLog(f"Error: {e}")

    def pause_process(self):
        try:
           
            start_program = "pause\n"
            self.robot_socket.sendall(start_program.encode())
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            
            self.insertLog(f"{data}")
            self.insertLog("Process paused.")
        except Exception as e:
            self.insertLog(f"Error: {e}")

    def stop_process(self):
        try:
           
            start_program = "stop\n"
            self.robot_socket.sendall(start_program.encode())
            data = self.robot_socket.recv(1024)
            data = data.decode('utf-8').strip()
            
            self.insertLog(f"{data}")
            self.insertLog("Process stopped.")
        except Exception as e:
            self.insertLog(f"Error: {e}")

    def z_filter(self):
        z_tolerance = 0.029
        points_temp = self.points
        filtered_points = []
        z_temp = []
        for point in points_temp:
            z_temp.append(point[2])
        z_max = min(z_temp)
        for point in points_temp:
            if (z_max - z_tolerance) <= point[2] <= (z_max + z_tolerance):
                filtered_points.append(point)
        # self.points = filtered_points  
        self.points = sorted(filtered_points, key=lambda coord: abs(coord[0]))  
  
    def update_camera_feed(self):
            while self.running:
                
                # Configure Realsense camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                color_image = np.asanyarray(color_frame.get_data())
                depth_sensor = self.profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()

                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame:
                    continue
                
                
                # Detect objects using the YOLO model
                results = self.model(color_image, conf=0.65, iou=0.7)
                annotated_frame = color_image.copy()
                annotated_frame = results[0].plot()
                
                obbs = results[0].obb
                if obbs is not None:
                    print(len(obbs))
                    for obb in obbs.data.cpu().numpy():
                        if len(obb) == 7:
                            x_center, y_center, width, height, angle, confidence, class_id = obb
                            if confidence >= 0.55:
                                if True:
                                    
                                    if self.PrecissionRequest == False:
                                        
                                        corners = self.get_obb_corners(x_center, y_center, width, height, angle)
                                        cornerList = []

                                        # Collect y-coordinates along with their index
                                        for i, corner in enumerate(corners):
                                            cornerList.append((corner[1], i))

                                        # Sort the corners based on their y-coordinates (ascending)
                                        sorted_corners = sorted(cornerList, key=lambda x: x[0])

                                        # Get the three bottom-most corners
                                        bottom_three = sorted_corners[-3:]

                                        # Extract the most bottom point
                                        bottom_point_index = bottom_three[-1][1]
                                        bottom_point = corners[bottom_point_index]

                                        # Draw the bottom-most point
                                        cv2.circle(annotated_frame, (int(bottom_point[0]), int(bottom_point[1])), 5, (255, 0, 0), -1)

                                        # Calculate distances and store the points
                                        distances = []
                                        for y_coord, index in bottom_three[:-1]:  # Skip the bottom-most point
                                            current_point = corners[index]

                                            # Calculate the distance
                                            distance = np.sqrt((current_point[0] - bottom_point[0]) ** 2 + (current_point[1] - bottom_point[1]) ** 2)
                                            distances.append((distance, current_point))

                                        # Sort distances to get the longest side
                                        distances.sort(reverse=True, key=lambda x: x[0])
                                        longest_side_distance, longest_side_point = distances[0]

                                        # Calculate the angle for the longest side
                                        dx = longest_side_point[0] - bottom_point[0]
                                        dy = longest_side_point[1] - bottom_point[1]
                                        angle = math.degrees(math.atan2(dy, dx))  # Angle in degrees
                                        if angle < 0: angle *=-1
                                        # Display the longest side and its angle
                                        cv2.line(
                                            annotated_frame,
                                            (int(bottom_point[0]), int(bottom_point[1])),
                                            (int(longest_side_point[0]), int(longest_side_point[1])),
                                            (0, 255, 0),
                                            2
                                        )

                                        # print(repr(brick_rotation))
                                        cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                                        

                                        # Display the simulated coordinates and class name
                                        class_name = results[0].names[int(class_id)]
                                        print(class_name)
                                        depth = depth_frame.get_distance(int(x_center), int(y_center))
                                        depth_value = depth_frame.get_distance(int(x_center), int(y_center)) / depth_scale
                                        depth_value = depth_value / 1000
                                        if 0 <= x_center <= 640 and 0 <= y_center <= 480:
                                            # Calculate 3D coordinates
                                            depth_min = 0.11 #meter
                                            depth_max = 1.0 #meter
                                            depth_intrinsic = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
                                            color_intrinsic = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

                                            depth_to_color_extrinsic =  self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.color))
                                            color_to_depth_extrinsic =  self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to( self.profile.get_stream(rs.stream.depth))

                                            color_pixel = (int(x_center), int(y_center))

                                            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(depth_frame.get_data(), depth_scale, 
                                                                depth_min, depth_max, depth_intrinsic, color_intrinsic, 
                                                                depth_to_color_extrinsic, color_to_depth_extrinsic, 
                                                                color_pixel)
                                            x_depth_pixel, y_depth_pixel = depth_pixel
                                            try:
                                                depth = depth_frame.get_distance(int(x_depth_pixel), int(y_depth_pixel))
                                            except:
                                                pass
                                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsic, [int(x_depth_pixel), int(y_depth_pixel)], depth)
                                            x_depth_point, y_depth_point, z_depth_point = depth_point 
                                            
                                            # Offset from camera location to tool location
                                            calculated_y = x_depth_point - (5.4/100) 
                                            calculated_x = y_depth_point - (5.5/100)
                                            camera_coords = np.array([calculated_x, calculated_y, 1])
                                            R = np.array([[1, 0, 0], 
                                                        [0, 1, 0], 
                                                        [0, 0, 1]])  
                                            T = np.array([-0.15170, -0.54743, 0])  # Robot position X,Y

                                            # Transformation: robot_coords = R @ camera_coords + T
                                            robot_coordinates = np.dot(R, camera_coords) + T

                                            cv2.putText(
                                                annotated_frame,
                                                f"X: {robot_coordinates[0]:.5f} | Y: {robot_coordinates[1]:.5f}",
                                                (int((longest_side_point[0] + bottom_point[0]) ), int((longest_side_point[1] + bottom_point[1]) / 2) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (255, 255, 0),
                                                1
                                            )
                                        
                                        if class_name == "brick":
                                            class_type = 1
                                        if class_name == "brick-side":
                                            class_type = 2 
                                        if robot_coordinates[0] is not None and self.PointRequest and len(self.points) <= len(obbs):
                                            angle_converted = angle
                                            # angle_counted = angleOffset(angle_converted)
                                            angle_counted = self.newAngleCounter(angle_converted)
                                            self.insertLog(f"x_center:{robot_coordinates[0]}, y_center:{robot_coordinates[1]}, depth:{depth_value}, angle:{int(angle_counted)}, class:{class_name}")
                                            self.points.append([robot_coordinates[0],robot_coordinates[1], depth_value, int(angle_counted), class_type, depth_value])
                                            if len(self.points) == len(obbs):
                                                self.z_filter()
                                                self.insertLog(f"Filtered points: {len(self.points)}")
                                                self.PointRequest = False 
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                self.recording.write(annotated_frame)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
                time.sleep(0.03)  # 30 FPS
    def insertLog(self, message):
        with open("test.txt", "a") as myfile:
            myfile.write(f"{datetime.datetime.now()} | {message}\n")
        self.log_listbox.insert(0, f"{datetime.datetime.now()} | {message}")
    def socket_server(self):
        while self.running:
            self.insertLog("Socket Server Started")
            while True:
                client_socket, client_address = self.server_socket.accept()
                self.insertLog(f"Connection established with {client_address}")
                try:
                    while True:
                        data = client_socket.recv(1024)
                        if not data:
                            self.insertLog(f"Connection closed by {client_address}")
                            break
                        recv = struct.unpack('>I', data)[0]
                        self.insertLog(f"Received: {recv}")
                        if recv == 99991: 
                            # Detection request
                            self.points = []
                            self.PointRequest = True
                            while self.PointRequest:
                                time.sleep(0.1)
                            client_socket.send(str(f"({len(self.points)})").encode())
                        if recv < 99991:
                            # coordinates request
                            if len(self.points) == 0:
                                client_socket.sendall(str(f"()").encode())
                                break
                            index = recv
                            print(f"Request points: {index}")
                            print(repr(self.points))
                            self.PrecissionId = recv 
                            points = self.points[int(index)]
                            print(f"x: {points[0]}, y: {points[1]}, height: {points[2]}, radians: {math.radians(points[3])}, type: {points[4]}")
                            client_socket.sendall(str(f"({points[0]}, {points[1]}, {(points[2])}, {(points[3])}, {(points[4])})").encode())
                                
                except Exception as e:
                    self.insertLog(f"Error: {e}")

                finally:
                    # Clean up the connection
                    client_socket.close()

    def angleOffset(self, Angle):
        if Angle < 0 :
            Angle * -1
        if Angle >= 85 and Angle <= 180 :
            return Angle
        if Angle <85 and Angle >=60:
            return Angle + 90
        if Angle < 60 and Angle >= 0:
            return 180-Angle 
    def get_obb_corners(self, x_center, y_center, width, height, angle):
        """
        Calculate the four corners of an Oriented Bounding Box (OBB).
        :param x_center: X-coordinate of the OBB center.
        :param y_center: Y-coordinate of the OBB center.
        :param width: Width of the OBB.
        :param height: Height of the OBB.
        :param angle: Rotation angle of the OBB in radians.
        :return: List of (x, y) tuples for the four corners.
        """
        # Calculate half dimensions
        half_width = width / 2
        half_height = height / 2
        
        # Rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Define the corner points relative to the center
        corners = np.array([
            [-half_width, -half_height],  # Bottom-left
            [ half_width, -half_height],  # Bottom-right
            [ half_width,  half_height],  # Top-right
            [-half_width,  half_height],  # Top-left
        ])
        
        # Rotate and translate the corners to the global coordinate system
        rotated_corners = np.dot(corners, rotation_matrix.T) + [x_center, y_center]
        
        return rotated_corners

    def newAngleCounter(self, Angle):
        """
        Calculate limited angle for the head rotation
        Parameters:
            Angle (integer): Angle in degrees

        Returns:
            int: angle
        """
        if Angle < 65:
            return Angle + 180
        else:
            return Angle
    def on_closing(self):
        self.recording.release()
        self.running = False
        
        try:
            self.pipeline.stop()
        except:
            pass
        self.root.destroy()
        self.server_socket.close()  # Add this line
        


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
