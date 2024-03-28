import math
import sys
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from numpy.typing import NDArray

from camera import Camera


Seconds = float
Image = NDArray
FloatVector = NDArray[np.float64]


SIMULATION_STEP_TIME: Seconds = 1/240
TARGET_TIME: Seconds = 5
DATA_PATH: str = "./Task_3/data"
GRAVITY: float = -10
CAMERA_VELOCITY: FloatVector = np.array([0, 0.1, 0])


def initClient(mode):
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

def get_camera_image(camera: Camera) -> Image:
    data = camera.get_frame()
    return cv2.UMat(np.asarray(data[:, :, [2, 1, 0]]))

def detect_aruco(img: Image) -> Image:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, marker_ids, rejected_candidates = detector.detectMarkers(img)
    #print("markerIds: ", (corners, np.array(marker_ids)))
    return corners, marker_ids

def render_aruco_markers(img: Image, corners, marker_ids):
    cv2.aruco.drawDetectedMarkers(img, corners, marker_ids)
    cv2.imshow('test', img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def main():
    initClient(p.GUI)
    _setupScene()

    camera = Camera()
    img = get_camera_image(camera)
    corners, marker_ids = detect_aruco(img)
    render_aruco_markers(img, corners, marker_ids)

    # # markerLength = 1
    # # distCoeffs = np.zeros([0,0,0,0])
    # # cameraMatrix = np.array([[1,0,100],[0,1,100],[0,0,1]])
    # #rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(	corners, markerLength, cameraMatrix, distCoeffs)

    current_time: Seconds = 0
    target_marker_id: int = 2
    marker_positions: list[float] = []
    while current_time <= TARGET_TIME:
        camera.move(CAMERA_VELOCITY)
        p.stepSimulation()

        img = get_camera_image(camera)
        corners, _ = detect_aruco(img)

        current_time += SIMULATION_STEP_TIME
        time.sleep(SIMULATION_STEP_TIME)

    p.disconnect()

    
def _setupScene() -> tuple[int, int]:
    p.resetSimulation()

    p.setGravity(0, 0, GRAVITY)
    p.setTimeStep(SIMULATION_STEP_TIME)

    planeId = 0#p.loadURDF("plane.urdf")
    boxId = p.loadURDF(f"{DATA_PATH}/simple.urdf.xml", useFixedBase=True)

    return planeId, boxId


if __name__ == "__main__":
    main()