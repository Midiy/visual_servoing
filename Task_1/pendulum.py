from scipy.integrate import odeint
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

from pe_typing import Seconds, Radians, FloatArray
from models import ControlMode, SceneInfo, SimulationResult, TheoreticalResult
from pid import PID, PIDComponents


# TASKS FROM 01.03.2024:
# 1  set maxTime
# 2  plots (pos, vel)
# 3  position control based on p.VELOCITY_CONTROL (proportional regulator)
# 4  position control based on p.TORQUE_CONTROL (PI-regulator)
# 5* compare plots of pybullet and our own odeint and figure out the source of errors and fix it
# 6* figure out how to add control to our own integration script

SIMULATION_TIME: Seconds = 5
SIMULATION_STEP_TIME: Seconds = 1/240   # pybullet simulation step
TARGET_JOINT_INDEX: int = 1
PENDULUM_LENGTH: float = 0.8
START_POSITION: Radians = 1             # starting position (radian)
TARGET_POSITION: Radians = -0.5
PROPORTIONAL_COEFF: float = 1
INTEGRAL_COEFF: float = 1
DIFFERENTIAL_COEFF: float = 1
GRAVITY: float = -10
DATA_PATH: str = "./Task_1/data"


def simulate(controller: PID, controlMode: ControlMode = ControlMode.FREE_MOTION, shouldWait: bool = False) -> SimulationResult:
    scene: SceneInfo = _setupScene()

    stepsCount: int = round(SIMULATION_TIME / SIMULATION_STEP_TIME)

    positions: FloatArray = np.zeros(stepsCount + 1)
    velocities: FloatArray = np.zeros(stepsCount + 1)

    targetPositionSin: np.float64 = np.sin(TARGET_POSITION)
    for step in range(stepsCount):
        jointPosition, jointVelocity, _, _ = p.getJointState(scene.boxId, jointIndex=TARGET_JOINT_INDEX)
        positions[step] = jointPosition
        velocities[step] = jointVelocity

        currentPositionSin: np.float64 = np.sin(jointPosition)
        control: np.float64 = controller.get_control(targetPositionSin, currentPositionSin) if controller else 0
        match controlMode:
            case ControlMode.FREE_MOTION:
                pass
            case ControlMode.VELOCITY_CONTROL:
                targetVelocity: np.float64 = control
                p.setJointMotorControl2(bodyIndex=scene.boxId, jointIndex=TARGET_JOINT_INDEX, targetVelocity=targetVelocity, controlMode=p.VELOCITY_CONTROL)
            case ControlMode.TORQURE_CONTROL:
                targetForce: np.float64 = control * PENDULUM_LENGTH * (-GRAVITY)
                p.setJointMotorControl2(bodyIndex=scene.boxId, jointIndex=TARGET_JOINT_INDEX, force=targetForce, controlMode=p.TORQUE_CONTROL)

        p.stepSimulation()
        if shouldWait:
            time.sleep(SIMULATION_STEP_TIME)

    jointPosition, jointVelocity, _, _ = p.getJointState(scene.boxId, jointIndex=TARGET_JOINT_INDEX)
    positions[-1] = jointPosition
    velocities[-1] = jointVelocity

    print(f"Final position: {jointPosition}, delta: {TARGET_POSITION - jointPosition}")

    return SimulationResult(positions, velocities)

def solveODE(timeSteps: FloatArray) -> TheoreticalResult:

    def rp(x: FloatArray, _):
        position: np.float64 = x[0]
        velocity: np.float64 = x[1]
        acceleration: np.float64 = GRAVITY / PENDULUM_LENGTH * np.sin(position)

        return [velocity, acceleration]

    result: FloatArray = odeint(rp, np.array([START_POSITION, 0], dtype=np.float64), timeSteps)
    positions: FloatArray = result[:, 0]
    velocities: FloatArray = result[:, 1]

    return TheoreticalResult(positions, velocities)

def initClient(mode):
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

def compareSimulationWithOde():
    initClient(p.DIRECT)

    # Additional step due to first values in results being initial position and velocity.
    stepsCount: int = round(SIMULATION_TIME / SIMULATION_STEP_TIME) + 1
    timeSteps: FloatArray = np.linspace(0.0, SIMULATION_TIME, stepsCount, dtype=np.float64)

    simulationResult: SimulationResult = simulate(controller=None)
    theoreticalResult: TheoreticalResult = solveODE(timeSteps)

    _, axs = plt.subplots(1, 2)
    _plotData(axs, timeSteps, simulationResult, theoreticalResult)
    plt.show()

    p.disconnect()

def withControl():
    initClient(p.GUI)
    _waitInput()

    # Additional step due to first values in results being initial position and velocity.
    stepsCount: int = round(SIMULATION_TIME / SIMULATION_STEP_TIME) + 1
    timeSteps: FloatArray = np.linspace(0.0, SIMULATION_TIME, stepsCount, dtype=np.float64)

    # Proportional regulator.
    controller = PID(SIMULATION_STEP_TIME, PIDComponents.PROPORTIONAL)
    velocitySimulationResult = simulate(controller, ControlMode.VELOCITY_CONTROL, shouldWait=True)

    _waitInput()

    # PID-regulator.
    controller = PID(SIMULATION_STEP_TIME, PIDComponents.PID)
    torqueSimulationResult = simulate(controller, ControlMode.TORQURE_CONTROL, shouldWait=True)

    _, axs = plt.subplots(2, 2)
    _plotData(axs[0], timeSteps, velocitySimulationResult, theoreticalResult=None, addTargetPosition=True)
    _plotData(axs[1], timeSteps, torqueSimulationResult, theoreticalResult=None, addTargetPosition=True)
    plt.show()

    p.disconnect()

def main():
    # Without control.
    compareSimulationWithOde()

    # Regulators.
    withControl()

def _setupScene() -> SceneInfo:
    p.resetSimulation()

    p.setGravity(0, 0, GRAVITY)
    p.setTimeStep(SIMULATION_STEP_TIME)

    planeId = p.loadURDF("plane.urdf")
    boxId = p.loadURDF(f"{DATA_PATH}/simple.urdf", useFixedBase=True)

    # get rid of all the default damping forces
    p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
    p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

    # go to the starting position
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=TARGET_JOINT_INDEX, targetPosition=START_POSITION, controlMode=p.POSITION_CONTROL)
    for _ in range(1000):
        p.stepSimulation()

    # turn off the motor for the free motion
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=TARGET_JOINT_INDEX, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

    return SceneInfo(planeId, boxId)

def _waitInput():
    _ = input()

def _plotData(axs,
              timeSteps: FloatArray, 
              simulationResult: SimulationResult, 
              theoreticalResult: TheoreticalResult, 
              addTargetPosition: bool = False):
    ax1, ax2 = axs[0], axs[1]
    
    ax1.plot(timeSteps, simulationResult.positions, label="Simulated position")
    if (theoreticalResult):
        ax1.plot(timeSteps, theoreticalResult.positions, label="Theoretical position")
    if addTargetPosition:
        ax1.plot(timeSteps, np.ones_like(timeSteps) * TARGET_POSITION)
    ax1.legend()

    ax2.plot(timeSteps, simulationResult.velocities, label="Simulated velocity")
    if (theoreticalResult):
        ax2.plot(timeSteps, theoreticalResult.velocities, label="Theoretical velocity")
    ax2.legend()
    

if __name__ == "__main__":
    main()