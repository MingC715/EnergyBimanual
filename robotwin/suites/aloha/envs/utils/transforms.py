from __future__ import annotations

import numpy as np
from pathlib import Path
import sapien.core as sapien
import transforms3d as t3d
from typing import Literal


import time
from functools import wraps


def rotate_cone(new_pt: np.ndarray, origin: np.ndarray, z_dir: np.ndarray = [0, 0, 1]):
    x = origin - new_pt
    x = x / np.linalg.norm(x)
    bx_ = np.array(z_dir).reshape(3)
    z = bx_ - np.dot(x, bx_) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _tolist(pose: sapien.Pose | list | np.ndarray) -> list:
    if isinstance(pose, list):
        return pose
    elif isinstance(pose, sapien.Pose):
        return pose.p.tolist() + pose.q.tolist()
    else:
        return pose.tolist()


def _toPose(pose: sapien.Pose | list | np.ndarray) -> sapien.Pose:
    if isinstance(pose, list):
        assert len(pose) == 7 or len(pose) == 3
        if len(pose) == 3:
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    elif isinstance(pose, np.ndarray):
        assert pose.shape == (7, ) or pose.shape == (3, )
        if pose.shape == (3, ):
            return sapien.Pose(pose[:3], [1, 0, 0, 0])
        else:
            return sapien.Pose(pose[:3], pose[3:])
    else:
        return pose


def rotate_along_axis(
    target_pose,
    center_pose,
    axis,
    theta: float = np.pi / 2,
    axis_type: Literal["center", "target", "world"] = "center",
    towards=None,
    camera_face=None,
) -> list:
    target_pose, center_pose = _toPose(target_pose), _toPose(center_pose)
    if theta == 0:
        return target_pose.p.tolist() + target_pose.q.tolist()
    rotate_mat = t3d.axangles.axangle2mat(axis, theta)

    target_mat = target_pose.to_transformation_matrix()
    center_mat = center_pose.to_transformation_matrix()
    if axis_type == "center":
        world_axis = (center_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    elif axis_type == "target":
        world_axis = (target_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    else:
        world_axis = np.array(axis).reshape(3)

    rotate_mat = t3d.axangles.axangle2mat(world_axis, theta)
    p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p
    if towards is not None:
        towards = np.dot(p - center_pose.p, np.array(towards).reshape(3))
        if towards < 0:
            rotate_mat = t3d.axangles.axangle2mat(world_axis, -theta)
            p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p

    if camera_face is None:
        q = t3d.quaternions.mat2quat(rotate_mat @ target_mat[:3, :3])
    else:
        q = t3d.quaternions.mat2quat(rotate_cone(p, center_pose.p, camera_face))
    return p.tolist() + q.tolist()


def rotate2rob(target_pose, rob_pose, box_pose, theta: float = 0.5) -> list:
    target_pose, rob_pose, box_pose = (
        _toPose(target_pose),
        _toPose(rob_pose),
        _toPose(box_pose),
    )

    target_mat = target_pose.to_transformation_matrix()
    v1 = (target_mat[:3, :3] @ np.array([[1, 0, 0]]).T).reshape(3)
    v2 = box_pose.p - rob_pose.p
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    return rotate_along_axis(
        target_pose=target_pose,
        center_pose=box_pose,
        axis=axis,
        theta=angle * theta,
        axis_type="world",
        towards=-v2,
    )


def choose_dirct(block_mat, base_pose: sapien.Pose):
    pts = block_mat[:3, :3] @ np.array([[1, -1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]])
    dirts = np.sum(np.power(pts - base_pose.p.reshape(3, 1), 2), axis=0)
    return pts[:, np.argmin(dirts)] + block_mat[:3, 3]


def cal_quat_dis(quat1, quat2):
    qmult = t3d.quaternions.qmult
    qinv = t3d.quaternions.qinverse
    qnorm = t3d.quaternions.qnorm
    delta_quat = qmult(qinv(quat1), quat2)
    return 2 * np.arccos(np.fabs((delta_quat / qnorm(delta_quat))[0])) / np.pi


def get_align_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = np.array(v1).reshape(3)
    v2 = np.array(v2).reshape(3)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    else:
        return t3d.axangles.axangle2mat(axis, angle)


def generate_rotate_vectors(
    axis: Literal["x", "y", "z"] | np.ndarray | list,
    angle: np.ndarray | list | float,
    base: np.ndarray | sapien.Pose | list = None,
    vector: np.ndarray | list = [1, 0, 0],
) -> np.ndarray:
    if base is None:
        base = np.eye(4)
    else:
        base = _toPose(base).to_transformation_matrix()

    if isinstance(axis, str):
        if axis == "x":
            axis = np.array([1, 0, 0])
        elif axis == "y":
            axis = np.array([0, 1, 0])
        elif axis == "z":
            axis = np.array([0, 0, 1])
        else:
            raise ValueError("axis must be x, y or z")
    else:
        axis = np.array(axis).reshape(3)

    axis = (base[:3, :3] @ axis.reshape(3, 1)).reshape(3)
    vector = (base[:3, :3] @ np.array(vector).reshape(3, 1)).reshape(3)

    vector = np.array(vector).reshape((3, 1))
    angle = np.array(angle).flatten()
    rotate_mat = np.zeros((3, angle.shape[0]))
    for idx, a in enumerate(angle):
        rotate_mat[:, idx] = (t3d.axangles.axangle2mat(axis, a) @ vector).reshape(3)
    return rotate_mat


def get_product_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1 = np.array(v1).reshape(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array(v2).reshape(3)
    return np.dot(v1, v2) * v1


def get_place_pose(
    actor_pose: np.ndarray | sapien.Pose | list,
    target_pose: np.ndarray | sapien.Pose | list,
    constrain: Literal["free", "align"] = "free",
    align_axis: list[np.ndarray] | np.ndarray | list = None,
    actor_axis: np.ndarray | list = [1, 0, 0],
    actor_axis_type: Literal["actor", "world"] = "actor",
    z_transform: bool = True,
) -> list:
    actor_pose_mat = _toPose(actor_pose).to_transformation_matrix()
    target_pose_mat = _toPose(target_pose).to_transformation_matrix()

    actor_pose_mat[:3, 3] = target_pose_mat[:3, 3]

    target_x = target_pose_mat[:3, 0]
    target_y = target_pose_mat[:3, 1]
    target_z = target_pose_mat[:3, 2]

    actor2world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
    if z_transform:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, :3] @ actor2world[:3, 2], target_z)
    else:
        z_align_matrix = get_align_matrix(actor_pose_mat[:3, 2], target_z)
    actor_pose_mat[:3, :3] = z_align_matrix @ actor_pose_mat[:3, :3]

    if constrain == "align":
        if align_axis is None:
            align_axis = np.array(target_pose_mat[:3, :3] @ np.array([[1, 0, 0]]).T)
        elif isinstance(align_axis, list):
            align_axis = np.array(align_axis).reshape((-1, 3)).T
        else:
            align_axis = np.array(align_axis).reshape((3, -1))
        align_axis = align_axis / np.linalg.norm(align_axis, axis=0)

        if actor_axis_type == "actor":
            actor_axis = actor_pose_mat[:3, :3] @ np.array(actor_axis).reshape(3, 1)
        elif actor_axis_type == "world":
            actor_axis = np.array(actor_axis)
        closest_axis_id = np.argmax(actor_axis.reshape(3) @ align_axis)
        align_axis = align_axis[:, closest_axis_id]

        actor_axis_xOy = get_product_vector(target_x, actor_axis) + get_product_vector(target_y, actor_axis)
        align_axis_xOy = get_product_vector(target_x, align_axis) + get_product_vector(target_y, align_axis)
        align_mat_xOy = get_align_matrix(actor_axis_xOy, align_axis_xOy)
        actor_pose_mat[:3, :3] = align_mat_xOy @ actor_pose_mat[:3, :3]

    return (actor_pose_mat[:3, 3].tolist() + t3d.quaternions.mat2quat(actor_pose_mat[:3, :3]).tolist())


def get_face_prod(q, local_axis, target_axis):
    """
    get product of local_axis (under q world) and target_axis
    """
    q_mat = t3d.quaternions.quat2mat(q)
    face = q_mat @ np.array(local_axis).reshape(3, 1)
    face_prod = np.dot(face.reshape(3), np.array(target_axis))
    return face_prod
