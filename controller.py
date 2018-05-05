"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0


class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""
        self.z_k_p = 25.
        self.z_k_d = 10.
        self.x_k_p = 3.
        self.x_k_d = 5.
        self.y_k_p = 3.
        self.y_k_d = 5.
        self.k_p_roll = 6.0
        self.k_p_pitch = 6.0
        self.k_p_yaw = 4.0
        self.k_p_p = 20.0
        self.k_p_q = 20.0
        self.k_p_r = 10.0
        return

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]

        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                       (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)

        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                                 acceleration_ff=np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        x_target, y_target = local_position_cmd
        x_dot_target, y_dot_target = local_velocity_cmd
        x_actual, y_actual = local_position
        x_dot_actual, y_dot_actual = local_velocity
        x_dot_dot_target, y_dot_dot_target = acceleration_ff
        b_x_c = self.x_k_p * (x_target - x_actual) + self.x_k_d * (x_dot_target - x_dot_actual) + x_dot_dot_target
        b_y_c = self.y_k_p * (y_target - y_actual) + self.y_k_d * (y_dot_target - y_dot_actual) + y_dot_dot_target
        return np.array([b_x_c, b_y_c])

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude,
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """
        rot_mat = euler2RM(*attitude)
        c = acceleration_ff
        c += self.z_k_p * (altitude_cmd - altitude) + self.z_k_d * (vertical_velocity_cmd - vertical_velocity)
        c = (1. / rot_mat[2, 2]) * (c + GRAVITY)
        c *= DRONE_MASS_KG

        if c > MAX_THRUST:
            c = MAX_THRUST
        elif c < 0:
            c = 0

        return c

    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        if thrust_cmd > 0:
            b_x_c_target = -np.clip(acceleration_cmd[0] * DRONE_MASS_KG / thrust_cmd, -1, 1);
            b_y_c_target = -np.clip(acceleration_cmd[1] * DRONE_MASS_KG / thrust_cmd, -1, 1);
            rot_mat = euler2RM(attitude[0],attitude[1],attitude[2])
            b_x_c_dot = self.k_p_roll * (b_x_c_target - rot_mat[0, 2])
            b_y_c_dot = self.k_p_pitch * (b_y_c_target - rot_mat[1, 2])
            r_mat = np.array([[rot_mat[1, 0] / rot_mat[2, 2], -rot_mat[0, 0] / rot_mat[2, 2]],
                              [rot_mat[1, 1] / rot_mat[2, 2], -rot_mat[0, 1] / rot_mat[2, 2]]])
            b_dot_mat = np.array([b_x_c_dot, b_y_c_dot]).T
            res = np.matmul(r_mat, b_dot_mat)
        else:
            res =  [0.,0.]
        return res

    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        p_c, q_c, r_c = body_rate_cmd
        p_actual, q_actual, r_actual = body_rate
        u_bar_p = self.k_p_p * (p_c - p_actual) * MOI[0]
        u_bar_q = self.k_p_q * (q_c - q_actual) * MOI[1]
        u_bar_r = self.k_p_r * (r_c - r_actual) * MOI[2]

        u_bar_p = np.clip(u_bar_p, -MAX_TORQUE, MAX_TORQUE)
        u_bar_q = np.clip(u_bar_q, -MAX_TORQUE, MAX_TORQUE)
        u_bar_r = np.clip(u_bar_r, -MAX_TORQUE, MAX_TORQUE)

        return [u_bar_p, u_bar_q, u_bar_r]

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        yaw_error = yaw_cmd - yaw
        if yaw_error > np.pi:
            yaw_error = yaw_error - 2.0 * np.pi
        elif yaw_error < -np.pi:
            yaw_error = yaw_error + 2.0 * np.pi
        r_c = self.k_p_yaw * yaw_error
        return r_c
