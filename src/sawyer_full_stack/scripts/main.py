#!/usr/bin/env python
"""
Final Script for EE106A Final Project - Sawyer Robot with AR Tag-Based Pick and Place
Author: Siyuan Zhai 2024
Original Author: Chris Correa
"""

import sys
import numpy as np
import rospkg
import roslaunch
import tf2_ros
import tf.transformations as tf_trans
import tf2_geometry_msgs
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from paths.trajectories import LinearTrajectory, CircularTrajectory
from paths.paths import MotionPath
from paths.path_planner import PathPlanner
from controllers.controllers import (
    PIDJointVelocityController,
    FeedforwardJointVelocityController
)
from utils.utils import *

from trac_ik_python.trac_ik import IK
import json
import cv2

import rospy
import intera_interface
from moveit_msgs.msg import DisplayTrajectory, RobotState
from sawyer_pykdl import sawyer_kinematics
from intera_interface import gripper as robot_gripper

from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker

# ============================================
# Configuration Parameters
# ============================================

FIXED_Z_HEIGHT = 0.54  # Fixed height for approaching (meters)
FIXED_Z_PLACE = 0.54     # Fixed height for placing (meters)
Z_OFFSET = 0.0          # Additional Z-axis offset in meters (adjust as needed)
pick_height = 0.4
Y_OFFSET = 0.03
# Paths
POSITIONS_JSON_PATH = '/home/cc/ee106a/fa24/class/ee106a-acs/ros_workspaces/lab_7/src/sawyer_full_stack/scripts/Processed_Results/processed_info.json'  # **Update this path**

# AR Marker Configuration
AR_MARKER_IDS = [2]  # List of AR markers to use

# Bin Positions (Update as needed)
RECYCLE_BIN_POS = [0.224, -0.763, 0.6]      # Position of recycle bin at fixed height
NONRECYCLE_BIN_POS = [0.185, -0.398, 0.6]   # Position of non-recycle bin at fixed height
task = 'line'

# ============================================
# Gripper Controller Class
# ============================================

class RobustGripperController:
    """A robust controller for handling gripper operations with retries and status checks."""

    def __init__(self, limb='right', retries=3, timeout=5.0):
        self.gripper = robot_gripper.Gripper(limb)
        self.retries = retries
        self.timeout = timeout
        self._calibrate_gripper()

    def _calibrate_gripper(self):
        """Calibrates the gripper and handles calibration failures."""
        try:
            self.gripper.calibrate()
            rospy.sleep(2)  # Wait for calibration to complete
            rospy.loginfo("Gripper calibrated successfully.")
        except Exception as e:
            rospy.logerr(f"Gripper calibration failed: {e}")
            sys.exit(1)

    def is_ready(self):
        """Checks if the gripper is ready for a new command."""
        status = self.gripper.status()
        return status.error == 0 and not status.moving

    def set_position(self, position):
        """
        Sets the gripper to the desired position with retries and status checks.

        Parameters:
        - position (float): Desired gripper position between 0.0 (closed) and 1.0 (open).

        Returns:
        - bool: True if successful, False otherwise.
        """
        for attempt in range(1, self.retries + 1):
            if not self.is_ready():
                rospy.logwarn("Gripper not ready. Waiting...")
                rospy.sleep(1)
                continue

            try:
                self.gripper.set_position(position)
                start_time = rospy.Time.now()
                rate = rospy.Rate(10)  # 10 Hz

                while (rospy.Time.now() - start_time).to_sec() < self.timeout:
                    current_pos = self.gripper.status().position
                    if abs(current_pos - position) < 0.05:
                        rospy.loginfo(f"Gripper reached position {position}.")
                        return True
                    rate.sleep()

                rospy.logwarn(f"Attempt {attempt}: Gripper did not reach position {position} within timeout.")
            except Exception as e:
                rospy.logerr(f"Attempt {attempt}: Failed to set gripper position: {e}")

            rospy.sleep(1)  # Wait before retrying

        rospy.logerr(f"Failed to set gripper position {position} after {self.retries} attempts.")
        return False

    def open_gripper(self, position=1.0):
        """Opens the gripper to the specified position."""
        return self.set_position(position)

    def close_gripper(self):
        """Closes the gripper."""
        return self.set_position(0.0)

    def reset_gripper(self):
        """Resets the gripper."""
        try:
            self.gripper.reset()
            rospy.loginfo("Gripper reset successfully.")
        except Exception as e:
            rospy.logerr(f"Failed to reset gripper: {e}")

# ============================================
# Utility Functions
# ============================================

def get_camera_intrinsics():
    """Retrieves camera intrinsic parameters."""
    # Example intrinsic matrix values; update as needed
    K = [919.8157958984375, 0.0, 641.1676635742188,
         0.0, 920.5045166015625, 360.57855224609375,
         0.0, 0.0, 1.0]

    try:
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        rospy.loginfo(f"Camera Intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        return fx, fy, cx, cy
    except rospy.ROSException:
        rospy.logerr("Timeout while retrieving camera intrinsics.")
        sys.exit(1)
    except Exception as e:
        rospy.logerr(f"Error retrieving camera intrinsics: {e}")
        sys.exit(1)

def pixel_to_camera_frame(u, v, fx, fy, cx, cy, fixed_z):
    """
    Converts pixel coordinates to camera frame coordinates using a fixed Z-value.

    Parameters:
    - u (int): Pixel row index.
    - v (int): Pixel column index.
    - fx (float): Focal length in x.
    - fy (float): Focal length in y.
    - cx (float): Principal point x-coordinate.
    - cy (float): Principal point y-coordinate.
    - fixed_z (float): Fixed Z-value in meters.

    Returns:
    - numpy.ndarray: 4-element vector representing the point in camera frame.
    """
    x = (u - cx) * fixed_z / fx
    y = (v - cy) * fixed_z / fy
    z = fixed_z
    point_camera = np.array([x, y, z, 1.0])
    rospy.loginfo(f"Pixel to Camera Frame - Pixel: ({u}, {v}), Point Camera: {point_camera}")
    return point_camera

def transform_to_base(camera_pose, point_camera, z_offset=0.0):
    """
    Transforms a point from the camera frame to the base frame,
    flipping the Z-axis and applying an additional Z-axis offset.

    Parameters:
    - camera_pose (geometry_msgs.msg.Pose): Pose of the camera in the base frame.
    - point_camera (numpy.ndarray): 4-element vector in the camera frame.
    - z_offset (float): Additional offset along the Z-axis in meters.

    Returns:
    - numpy.ndarray: 3-element vector in the base frame, or None if transformation fails.
    """
    try:
        rospy.loginfo("Starting transformation from camera frame to base frame.")
        # Construct transformation matrix from camera_pose
        translation = [camera_pose.position.x, camera_pose.position.y, camera_pose.position.z]
        rotation = [camera_pose.orientation.x, camera_pose.orientation.y,
                    camera_pose.orientation.z, camera_pose.orientation.w]
        matrix = tf_trans.concatenate_matrices(
            tf_trans.translation_matrix(translation),
            tf_trans.quaternion_matrix(rotation)
        )
        rospy.loginfo(f"Camera Pose Matrix:\n{matrix}")

        # Define a rotation to flip the Z-axis (180 degrees around X-axis)
        flip_z_rotation = tf_trans.rotation_matrix(np.pi, [1, 0, 0])  # 180 degrees rotation around X-axis
        rospy.loginfo(f"Flip Z Rotation Matrix:\n{flip_z_rotation}")

        # Apply the flip to the transformation matrix
        adjusted_matrix = np.dot(matrix, flip_z_rotation)
        rospy.loginfo(f"Adjusted Transformation Matrix with Flip:\n{adjusted_matrix}")

        # Define additional translation for Z-offset
        z_offset_translation = tf_trans.translation_matrix([0, -0.03, 0])
        rospy.loginfo(f"Z-Offset Translation Matrix:\n{z_offset_translation}")

        # Apply the Z-offset translation
        final_matrix = np.dot(adjusted_matrix, z_offset_translation)
        rospy.loginfo(f"Final Transformation Matrix with Z-Offset:\n{final_matrix}")

        # Transform point to base frame with adjusted rotation and Z-offset
        point_base = np.dot(final_matrix, point_camera)
        rospy.loginfo(f"Transformed Point in Base Frame: {point_base[:3]}")

        # Validate transformed point
        if np.any(np.isnan(point_base)) or np.any(np.isinf(point_base)):
            rospy.logerr(f"Invalid transformed point: {point_base[:3]}")
            return None

        return point_base[:3]
    except Exception as e:
        rospy.logerr(f"Error in transform_to_base: {e}")
        return None

def tuck_arm():
    """
    Tucks the robot arm to the start position. Use with caution.
    """
    try:
        response = raw_input('Would you like to tuck the arm? (y/n): ')
    except NameError:
        response = input('Would you like to tuck the arm? (y/n): ')

    if response.lower() == 'y':
        rospack = rospkg.RosPack()
        try:
            path = rospack.get_path('sawyer_full_stack')
            launch_path = path + '/launch/custom_sawyer_tuck.launch'
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
            launch.start()
            rospy.loginfo("Tucking arm launched successfully.")
        except rospkg.ResourceNotFound:
            rospy.logerr("Launch file 'custom_sawyer_tuck.launch' not found.")
        except Exception as e:
            rospy.logerr(f"Failed to launch tuck_arm: {e}")
    else:
        rospy.loginfo('Canceled. Not tucking the arm.')

def lookup_tag(tag_number, tf_buffer):
    """
    Looks up the position and orientation of an AR tag.

    Parameters:
    - tag_number (int): ID of the AR tag.
    - tf_buffer (tf2_ros.Buffer): TF buffer for transformations.

    Returns:
    - tuple: (position as numpy array, rotation as list) or None if not found.
    """
    try:
        trans = tf_buffer.lookup_transform('base', f'ar_marker_{tag_number}', rospy.Time(0), rospy.Duration(10.0))
        rospy.loginfo(f"AR tag {tag_number} found.")
    except Exception as e:
        rospy.logwarn(f"AR tag {tag_number} not found: {e}")
        return None

    tag_pos = [getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')]
    rotation = [getattr(trans.transform.rotation, dim) for dim in ('x', 'y', 'z', 'w')]
    rospy.loginfo(f"AR Tag {tag_number} Position: {tag_pos}, Rotation: {rotation}")
    return np.array(tag_pos), rotation


def get_trajectory(tfBuffer, limb, kin, ik_solver, target_pos, num_way, task):
    """Returns a trajectory based on the task (line or circle)."""

    try:
        trans = tfBuffer.lookup_transform('base', 'right_hand', rospy.Time(0), rospy.Duration(10.0))
    except Exception as e:
        rospy.logerr(f"Error looking up right_hand transform: {e}")
        return None

    current_position = np.array([getattr(trans.transform.translation, dim) for dim in ('x', 'y', 'z')])
    rospy.loginfo(f"Current Position: {current_position}")

    if task == 'line':
        goal_position = target_pos.copy()
        goal_position[2] += 0.4  # Linear path moves to a Z position above AR Tag
        goal_position[0] += -0.05
        goal_position[1] += 0.14
        rospy.loginfo(f"TARGET POSITION for Line: {goal_position}")
        trajectory = LinearTrajectory(start_position=current_position, goal_position=goal_position, total_time=9)
    elif task == 'circle':
        center_position = target_pos.copy()
        center_position[2] += 0.5
        rospy.loginfo(f"TARGET POSITION for Circle: {center_position}")
        trajectory = CircularTrajectory(center_position=center_position, radius=0.1, total_time=15)
    else:
        rospy.logerr(f'Task "{task}" not recognized')
        return None

    path = MotionPath(limb, kin, ik_solver, trajectory)
    robot_traj = path.to_robot_trajectory(num_way, True)

    if robot_traj is None:
        rospy.logerr(f"Failed to generate trajectory for task '{task}'.")
    else:
        rospy.loginfo(f"Trajectory for task '{task}' generated successfully.")

    return robot_traj


def execute_trajectory(planner, pub, robot_trajectory):
    """
    Publishes and executes the given trajectory.

    Parameters:
    - planner (PathPlanner): Path planner instance.
    - pub (rospy.Publisher): Publisher for displaying the trajectory.
    - robot_trajectory (moveit_msgs.msg.RobotTrajectory): The trajectory to execute.

    Returns:
    - bool: True if execution was successful, False otherwise.
    """
    try:
        # Publish the trajectory for visualization
        disp_traj = DisplayTrajectory()
        disp_traj.trajectory.append(robot_trajectory)
        disp_traj.trajectory_start = RobotState()
        pub.publish(disp_traj)
        rospy.loginfo("Published trajectory for visualization.")
        # Extract joint positions from the last waypoint
        # Directly execute the precomputed linear trajectory
        # This will move along all intermediate waypoints instead of just the final pose
        success = planner.execute_plan(robot_trajectory)
        if not success:
            rospy.logerr("Failed to execute the given trajectory directly.")
            return False

        rospy.loginfo("Executed the given linear trajectory successfully.")
        return True
    except Exception as e:
        rospy.logerr(f"Failed to execute trajectory: {e}")
        return False
    
    #     joint_positions = robot_trajectory.joint_trajectory.points[-1].positions
    #     rospy.loginfo(f"Executing Joint Positions: {joint_positions}")

    #     # Plan to the target joint positions
    #     plan = planner.plan_to_joint_pos(joint_positions)
    #     if plan is None:
    #         rospy.logerr("Planning failed.")
    #         return False

    #     # Execute the plan
    #     planner.execute_plan(plan)
    #     rospy.loginfo("Executed trajectory successfully.")
    #     return True
    # except Exception as e:
    #     rospy.logerr(f"Failed to execute trajectory: {e}")
    #     return False

def load_and_validate_json(file_path):
    """
    Loads and validates the JSON file containing item positions and categories.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - list: List of items if successful, None otherwise.
    """
    try:
        with open(file_path, 'r') as f:
            # Read the entire file content
            content = f.read()
            rospy.loginfo(f"Raw file content length: {len(content)}")
            rospy.loginfo(f"Raw file content (first 500 chars): {content[:500]}")

            try:
                # Try standard JSON parsing
                items = json.loads(content)
                rospy.loginfo(f"Parsed JSON type: {type(items)}")
                rospy.loginfo(f"Parsed JSON content: {items}")

                # If it's a single dictionary, wrap it in a list
                if isinstance(items, dict):
                    items = [items]

                # Validate that we now have a list
                if not isinstance(items, list):
                    rospy.logerr(f"Expected a list, got {type(items)}")
                    return None

                return items

            except json.JSONDecodeError as e:
                rospy.logerr(f"JSON decoding error: {e}")
                rospy.logerr(f"Problematic content: {content}")
                return None

    except FileNotFoundError:
        rospy.logerr(f"Positions JSON file not found at {file_path}.")
        return None
    except Exception as e:
        rospy.logerr(f"Failed to load positions file: {e}")
        return None


def process_items(items, camera_pose, camera_rotation):
    """
    Processes each item from the JSON data to compute their 3D positions.

    Parameters:
    - items (list): List of items from the JSON file.
    - fx, fy, cx, cy (float): Camera intrinsic parameters.
    - camera_pose (numpy.ndarray): Position of the camera in base frame.
    - camera_rotation (list): Orientation of the camera in base frame.

    Returns:
    - tuple: (list of 3D positions, list of categories)
    """
    item_3d_positions = []
    item_categories = []
    fx, fy, cx, cy = get_camera_intrinsics()
    if not items:
        rospy.logerr("No items to process")
        return item_3d_positions, item_categories

    rospy.loginfo(f"Total items to process: {len(items)}")

    for index, item in enumerate(items):
        rospy.loginfo(f"Processing item {index}: {type(item)}")
        rospy.loginfo(f"Item content: {item}")

        try:
            # Ensure item is a dictionary
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception as parse_error:
                    rospy.logerr(f"Could not parse item {index} as JSON: {parse_error}")
                    continue

            # Validate item structure
            if not isinstance(item, dict):
                rospy.logerr(f"Item {index} is not a dictionary: {type(item)}")
                continue

            # Ensure required keys exist
            if 'coordinates' not in item or 'labels' not in item or 'categories' not in item:
                rospy.logerr(f"Item {index} missing required keys")
                continue

            # Extract coordinates and labels
            coordinates = item.get('coordinates', [])
            labels = item.get('labels', [])
            categories = item.get('categories', [])

            # Validate coordinates and labels
            if not coordinates or not labels or not categories:
                rospy.logerr(f"Invalid coordinates, labels, or categories in item {index}")
                continue

            # Extract first coordinate pair
            if isinstance(coordinates[0], list) or isinstance(coordinates[0], tuple):
                u, v = coordinates[0]  # Assuming the first coordinate pair
            else:
                u, v = coordinates  # If coordinates are flat

            category = labels[0] if labels else "Unknown"  # Assuming first label corresponds to category

            rospy.loginfo(f"Item {index} - Pixel Position: ({u}, {v}), Category: {category}")

            # Convert pixel to camera frame
            point_camera = pixel_to_camera_frame(u, v, fx, fy, cx, cy, FIXED_Z_HEIGHT)
            #FIXED_Z_HEIGHT = 0.54, dist from camera to object
            # Create Pose message for camera
            camera_pose_msg = Pose()
            camera_pose_msg.position.x, camera_pose_msg.position.y, camera_pose_msg.position.z = camera_pose
            camera_pose_msg.orientation.x, camera_pose_msg.orientation.y, camera_pose_msg.orientation.z, camera_pose_msg.orientation.w = camera_rotation

            # Transform to base frame
            point_base = transform_to_base(camera_pose_msg, point_camera, z_offset=Z_OFFSET)

            if point_base is None:
                rospy.logerr(f"Transformation failed for item {index}. Skipping.")
                continue

            item_3d_positions.append(point_base)
            item_categories.append(category)

        except Exception as e:
            rospy.logerr(f"Error processing item {index}: {e}")

    if not item_3d_positions:
        rospy.logwarn("No items processed from JSON.")

    return item_3d_positions, item_categories



def main():
    """Main function to execute the trajectory and control the robot."""
    # Configuration Parameters
    ar_marker = AR_MARKER_IDS  # Which AR markers to use
    num_way = 50  # Number of waypoints in trajectory

    # Initialize ROS Node
    rospy.init_node('moveit_node', anonymous=True)
    rospy.loginfo("ROS node 'moveit_node' initialized.")

    # Initialize Marker Publisher for RViz Visualization
    marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rospy.sleep(1)  # Wait for publisher to connect

    # Tuck the arm (optional)
    tuck_arm()

    # Initialize TF Buffer and Listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1)  # Wait for TF buffer to fill
    rospy.loginfo("TF listener initialized.")

    # Lookup AR Tag Positions
    tag_pos = []
    for marker in ar_marker:
        result = lookup_tag(marker, tf_buffer)
        if result is not None:
            tag_pos.append(result)
        else:
            rospy.logwarn(f"AR tag {marker} not found.")

    if not tag_pos:
        rospy.logerr("No AR tags found. Exiting.")
        sys.exit(1)

    camera_pose, camera_rotation = tag_pos[0]
    rospy.loginfo(f"Camera Pose: {camera_pose}")
    rospy.loginfo(f"Camera Rotation: {camera_rotation}")

    # Define Bin Positions
    recycle_bin = RECYCLE_BIN_POS
    nonrecycle_bin = NONRECYCLE_BIN_POS


    # Initialize Gripper Controller
    gripper_controller = RobustGripperController(limb='right', retries=3, timeout=5.0)

    # Initialize Kinematics and Planner

    ik_solver = IK("base", "right_gripper_tip")

    limb_interface = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")
    planner = PathPlanner('right_arm')
    traj_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=10)
    rospy.sleep(1)  # Wait for publisher to connect
    rospy.loginfo("Initialized kinematics, planner, and trajectory publisher.")

    # Initialize RViz Marker Publisher
    # Already initialized earlier as marker_pub

    # Process Each Item
    items = load_and_validate_json(POSITIONS_JSON_PATH)

    if items is None:
        rospy.logerr("Failed to load items")
        sys.exit(1)

    # Process items with necessary parameters
    item_3d_positions, item_categories = process_items(
        items, camera_pose, camera_rotation
    )

    rospy.loginfo(f"3D Positions: {item_3d_positions}")
    rospy.loginfo(f"Categories: {item_categories}")


    # Iterate Through Each Item for Pick and Place
    for idx, (item_pos, category) in enumerate(zip(item_3d_positions, item_categories), 1):
        rospy.loginfo(f"Starting pick and place for item {idx}: Category={category}, Position={item_pos}")
        before_pickup_pos = item_pos.copy()
        before_pickup_pos[2] += pick_height
        # 1. Approach Trajectory to Fixed Height Above Item
        approach_trajectory = get_trajectory(
            tf_buffer, limb_interface, kin, ik_solver, before_pickup_pos, num_way, task)
        if approach_trajectory is None:
            rospy.logerr("Failed to generate approach trajectory. Skipping item.")
            continue

        if not execute_trajectory(planner, traj_pub, approach_trajectory):
            rospy.logerr("Failed to execute approach trajectory. Skipping item.")
            continue

        # 2. Pickup Trajectory - Move Down to Item
        pickup_trajectory = get_trajectory(
            tf_buffer,
            limb=limb_interface,
            kin=kin,
            ik_solver=ik_solver,
            target_pos=item_pos,
            num_way=num_way,
            task=task
        )
        if pickup_trajectory is None:
            rospy.logerr("Failed to generate pickup trajectory. Skipping item.")
            continue

        if not execute_trajectory(planner, traj_pub, pickup_trajectory):
            rospy.logerr("Failed to execute pickup trajectory. Skipping item.")
            continue

        # 3. Close Gripper to Grab Item
        if not gripper_controller.close_gripper():
            rospy.logerr("Failed to close gripper after pickup. Skipping item.")
            continue

        # 4. Lift Trajectory - Move Back to Approach Height with Item

        lift_trajectory = get_trajectory(
            tf_buffer,
            limb=limb_interface,
            kin=kin,
            ik_solver=ik_solver,
            target_pos=before_pickup_pos,
            num_way=num_way,
            task=task
        )
        if lift_trajectory is None:
            rospy.logerr("Failed to generate lift trajectory. Skipping item.")
            continue

        if not execute_trajectory(planner, traj_pub, lift_trajectory):
            rospy.logerr("Failed to execute lift trajectory. Skipping item.")
            continue

        # 5. Transit Trajectory - Move to Fixed Height Above Target Bin
        target_bin = recycle_bin if category.lower() == "recyclable" else nonrecycle_bin
        target_bin_xy = target_bin[:2]

        transit_trajectory = get_trajectory(
            tf_buffer,
            limb=limb_interface,
            kin=kin,
            ik_solver=ik_solver,
            target_pos=[target_bin_xy[0], target_bin_xy[1], FIXED_Z_HEIGHT],
            num_way=num_way,
            task=task
        )
        if transit_trajectory is None:
            rospy.logerr("Failed to generate transit trajectory. Skipping item.")
            continue

        if not execute_trajectory(planner, traj_pub, transit_trajectory):
            rospy.logerr("Failed to execute transit trajectory. Skipping item.")
            continue

        # 6. Place Trajectory - Move Down to Place Position
        place_pos = target_bin.copy()
        place_pos[2] -= 0.05  # Adjust to move down into the bin; change as needed

        place_trajectory = get_trajectory(
            tf_buffer,
            limb=limb_interface,
            kin=kin,
            ik_solver=ik_solver,
            target_pos=place_pos,
            num_way=num_way,
            task=task
        )
        if place_trajectory is None:
            rospy.logerr("Failed to generate place trajectory. Skipping item.")
            continue

        if not execute_trajectory(planner, traj_pub, place_trajectory):
            rospy.logerr("Failed to execute place trajectory. Skipping item.")
            continue

        # 7. Open Gripper to Release the Item
        if not gripper_controller.open_gripper(position=1.0):
            rospy.logerr("Failed to open gripper after placing.")
            continue

        # 8. Reset Gripper (if necessary)
        gripper_controller.reset_gripper()

        rospy.loginfo(f"Pick and place operation completed for item {idx}.")

    rospy.loginfo("All pick and place operations completed successfully.")

    rospy.spin()  # Keep the script alive until manually terminated

# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

# ============================================
# Additional Instructions
# ============================================

# To ensure RGB and depth alignment, execute the following commands:
# 1. Check image dimensions:
#    rostopic echo /camera/color/image_raw | grep -E 'height|width'
#    rostopic echo /camera/depth/image_rect_raw | grep -E 'height|width'

# 2. Launch Realsense Camera:
#    roslaunch realsense2_camera rs_camera.launch

# 3. Verify Realsense Topics:
#    rostopic list | grep realsense

# 4. View Images:
#    rosrun rqt_image_view rqt_image_view

# C