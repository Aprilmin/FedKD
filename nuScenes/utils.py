import numpy as np
from scipy.integrate import cumulative_trapezoid
import cv2
from pyquaternion import Quaternion
import re
from math import atan2
import torch
import random
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def obatainValueByText(waypoints, fut_start_world, obs_ego_velocities, return_all=True):
    # coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", waypoints)
    # speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
    # speed_curvature_pred = speed_curvature_pred[:10]
    try:
        nums = re.findall(r'\[\s*[\d\.\-eE]+\s*,\s*[\d\.\-eE]+\s*\]', waypoints)
        speed_curvature_pred = [eval(n) for n in nums[:10]]
        # print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

        pred_len = len(speed_curvature_pred)
        curvatures = np.array(speed_curvature_pred)[:, 1] / 100
        speeds = np.array(speed_curvature_pred)[:, 0]
        traj = np.zeros((pred_len, 3))  # 轨迹（x,y,z）
        traj[:pred_len, :2] = IntegrateCurvatureForPoints(curvatures, speeds, fut_start_world,
                                                          atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                                                          pred_len)
        if return_all:
            return curvatures, speeds, traj
        else:
            return traj
    except:
        return None
def IntegrateCurvatureForPoints(curvatures, velocities_norm, initial_position, initial_heading, time_span):
    t = np.linspace(0, time_span, time_span)  # Time vector

    # Initial conditions
    x0, y0 = initial_position[0], initial_position[1]  # Starting position
    theta0 = initial_heading  # Initial orientation (radians)

    # Integrate to compute heading (theta)
    theta = cumulative_trapezoid(curvatures * velocities_norm, t, initial=theta0)
    theta[1:] += theta0

    # Compute velocity components
    v_x = velocities_norm * np.cos(theta)
    v_y = velocities_norm * np.sin(theta)

    # Integrate to compute trajectory
    x = cumulative_trapezoid(v_x, t, initial=x0)
    y = cumulative_trapezoid(v_y, t, initial=y0)

    x[1:] += x0
    y[1:] += y0

    return np.stack((x, y), axis=1)
def TransformPoint(point, transform):
    """ Transform a 3D point using a transformation matrix. """
    if isinstance(point, list):
        point = np.array(point)

    if point.shape[-1] == 3:
        point = np.append(point, 1)
    transformed_point = transform @ point
    return transformed_point[:3]
def OffsetTrajectory3D(points, offset_distance):
    """
    Offsets a 3D trajectory by a specified distance normal to the trajectory.

    Parameters:
        points (np.ndarray): n x 3 array representing the 3D trajectory (x, y, z).
        offset_distance (float): Distance to offset the trajectory.

    Returns:
        np.ndarray: Offset trajectory as an n x 3 array.
    """
    # Compute differences to find tangent vectors
    tangents = np.gradient(points, axis=0)  # Approximate tangents
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)  # Normalize tangents

    # Reference vector for normal plane computation (e.g., z-axis)
    reference_vector = np.array([0, 0, 1])

    # Compute normal vectors via cross product
    normals = np.cross(tangents, reference_vector)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize normals

    # Compute offset points
    offset_points = points + offset_distance * normals

    return offset_points
def ProjectWorldToImage(points3d_world: list, cam_to_ego, ego_to_world):
    # Plot the waypoints.

    T_ego_global = FormTransformationMatrix(ego_to_world['translation'], Quaternion(ego_to_world['rotation']))
    T_cam_ego = FormTransformationMatrix(cam_to_ego['translation'], Quaternion(cam_to_ego['rotation']))
    T_cam_global = T_ego_global @ T_cam_ego
    T_global_cam = np.linalg.inv(T_cam_global)

    points3d_cam = [TransformPoint(point, T_global_cam) for point in points3d_world]

    points3d_img = ProjectEgoToImage(np.array(points3d_cam), cam_to_ego['camera_intrinsic'])

    return points3d_img
def ProjectEgoToImage(points_3d: np.array, K):
    """ Project 3D points to 2D using camera intrinsic matrix K. """
    # Filter out points that are behind the camera
    points_3d = points_3d[points_3d[:, 2] > 0]

    # Project the remaining points
    points_2d = np.dot(K, points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2][:, np.newaxis]  # Normalize by depth
    return points_2d
def OverlayTrajectory(img, points3d_world: list, cam_to_ego, ego_to_world, color=(0, 0, 255), plot_fig=True):

    # Construct left/right boundaries.
    points3d_left_world = OffsetTrajectory3D(np.array(points3d_world), -1.73 / 2)
    points3d_right_world = OffsetTrajectory3D(np.array(points3d_world), 1.73 / 2)

    # Project the waypoints to the image.
    points3d_img = ProjectWorldToImage(points3d_world, cam_to_ego, ego_to_world)
    points3d_left_img = ProjectWorldToImage(points3d_left_world.tolist(), cam_to_ego, ego_to_world)
    points3d_right_img = ProjectWorldToImage(points3d_right_world.tolist(), cam_to_ego, ego_to_world)

    if plot_fig:
        # Overlay the waypoints on the image.
        for i in range(len(points3d_img) - 1):
            cv2.circle(img, tuple(points3d_img[i].astype(int)), radius=6, color=color, thickness=-1)

        # # Draw lines.
        # for i in range(len(points3d_img) - 1):
        #     cv2.line(img, tuple(points3d_img[i].astype(int)), tuple(points3d_img[i+1].astype(int)), color, 2)

    # Draw sweep area polygon between the boundaries.
    frame = np.zeros_like(img)
    polygon = np.vstack((np.array(points3d_left_img), np.array(points3d_right_img)[::-1])).astype(np.int32)
    check_flag = False
    if polygon.size == 0:
        check_flag = True
        return check_flag
    if plot_fig:
        cv2.fillPoly(frame, [polygon], color=color)  # Green polygon
        mask = frame.astype(bool)
        img[mask] = cv2.addWeighted(img, 0.5, frame, 0.5, 0)[mask]
    return check_flag

def FormTransformationMatrix(translation, rotation):
    """ Create a transformation matrix from translation and rotation (as a quaternion). """
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

