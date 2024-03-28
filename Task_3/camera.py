from typing import Any

import numpy as np
from numpy.typing import NDArray
import pybullet as pb


FloatVector = NDArray[np.float64]


class Camera:
    _DEFAULT_SIZE = {
        'width': 1000,
        'height': 1000
    }

    _size: dict[str, int]
    _camera_position: FloatVector
    _target_position: FloatVector
    _view_matrix: Any
    _projection_matrix: Any
    _cam_image_kwargs: dict[str, Any]

    def __init__(self,
                 camera_position: FloatVector = np.array([0, 0.5, 0.5]),
                 target_position: FloatVector = np.array([0, 0.5, 0.5]),
                 fov: int = 60,
                 size: dict[str, int] = None):
        if size is None:
            size = Camera._DEFAULT_SIZE
        self._size = size
        self._projection_matrix = pb.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=100)
        self._camera_position = camera_position
        self._target_position = target_position
        self.move(np.zeros(3))

    def move(self, move_vector: FloatVector):
        self._camera_position += move_vector
        self._target_position += move_vector

        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=self._camera_position,
            cameraTargetPosition=self._target_position,
            cameraUpVector=[0, 0, 1])
        self._cam_image_kwargs = {
            **self._size,
            'viewMatrix': view_matrix,
            'projectionMatrix': self._projection_matrix,
            'renderer': pb.ER_TINY_RENDERER
        }

    def get_frame(self):
        """
        returns RGBA array of size (x, y, 4)
        """
        return pb.getCameraImage(**self._cam_image_kwargs)[2]
