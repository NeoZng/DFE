"""Utils for colmap data processing.
"""

import math
import numpy as np


def get_cam(img, cams):
    """Get camera data.

    Args:
        img: image data
        cams: camera data

    Returns:
        tuple: intrinsic, extrinsic, image size
    """
    camera = cams[int(img.camera_id)]

    params = camera.params
    K = np.eye(3)
    K[0, 0] = params[0]
    K[1, 1] = params[1]
    K[0, 2] = params[2]
    K[1, 2] = params[3]

    return K

def quaterion_to_rotation_matrix(quaternion):
    """Get rotation matrix from quaternion.

    Args:
        quaternion (array): quaternion

    Returns:
        array: rotation matrix
    """

    #    Return homogeneous rotation matrix from quaternion.

    # >> > M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    # >> > numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    # True
    # >> > M = quaternion_matrix([1, 0, 0, 0])
    # >> > numpy.allclose(M, numpy.identity(4))
    # True
    # >> > M = quaternion_matrix([0, 1, 0, 0])
    # >> > numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    # True

    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)

    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def compute_residual(pts, F):
    """Compute epipolar residual.

    Args:
        pts (array): points
        F (array): fundamental matrix

    Returns:
        array: residuals
    """
    pts1 = points_to_homogeneous(pts[:, :2])
    pts2 = points_to_homogeneous(pts[:, 2:])

    line_2 = np.dot(F.T, pts1.T)
    line_1 = np.dot(F, pts2.T)

    dd = np.sum(line_2.T * pts2, 1)

    d = np.abs(dd) * (
        1.0 / np.sqrt(line_1[0, :] ** 2 + line_1[1, :] ** 2)
        + 1.0 / np.sqrt(line_2[0, :] ** 2 + line_2[1, :] ** 2)
    )

    return d


def vector_to_cross(vec):
    """Compute cross product matrix.

    Args:
        vec (array): vector

    Returns:
        array: cros product matrix
    """
    T = np.zeros((3, 3))

    T[0, 1] = -vec[2]
    T[0, 2] = vec[1]
    T[1, 0] = vec[2]
    T[1, 2] = -vec[0]
    T[2, 0] = -vec[1]
    T[2, 1] = vec[0]

    return T


def points_to_homogeneous(pts):
    """Transform points to homogeneous points.

    Args:
        pts (array): points

    Returns:
        array: homogeneous points
    """
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), 1)
