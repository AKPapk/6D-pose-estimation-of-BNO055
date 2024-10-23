import numpy as np

class PoseKF:
    def __init__(self, state_init, state_cov_init, process_cov, measurement_cov):
        """
        Initializes the Kalman Filter for pose estimation (quaternion + position + velocity).

        Args:
            state_init: Initial state vector [qx, qy, qz, qw, px, py, pz, vx, vy, vz]
            state_cov_init: Initial state covariance matrix (9x9) for uncertainty.
            process_cov: Process noise covariance matrix (9x9)
            measurement_cov: Measurement noise covariance matrix (9x9)
        """
        self.state = np.array(state_init)  # [qx, qy, qz, qw, px, py, pz, vx, vy, vz]
        self.state_cov = np.array(state_cov_init)  # 9x9 covariance matrix
        
        # Process and measurement noise covariance matrices
        self.process_cov = np.array(process_cov)
        self.measurement_cov = np.array(measurement_cov)
        
        # Identity matrix
        self.identity = np.eye(9)
        
    def predict(self, dt):
        """
        Predict the new state based on current state and time step (dt).

        Args:
            dt: Time step (delta time)
        """
        # Extract current quaternion (orientation), position, and velocity
        quat = self.state[0:4]  # [qx, qy, qz, qw]
        position = self.state[4:7]  # [px, py, pz]
        velocity = self.state[7:10]  # [vx, vy, vz]
        
        # Quaternion prediction remains the same (no rotational acceleration)
        new_quat = quat  # No change in quaternion during prediction step
        
        # Predict new position and velocity (simple constant velocity model)
        new_position = position + velocity * dt
        new_velocity = velocity  # No acceleration assumption in this simple model
        
        # Update state vector with predicted values
        self.state[0:4] = new_quat
        self.state[4:7] = new_position
        self.state[7:10] = new_velocity
        
        # Predict the new state covariance
        self.state_cov = self.state_cov + self.process_cov * dt
    
    def update(self, measurement):
        """
        Update the state and covariance using the measurement.

        Args:
            measurement: Measurement vector [qx, qy, qz, qw, px, py, pz, vx, vy, vz]
        """
        # Measurement residual (difference between measurement and predicted state)
        residual = np.array(measurement) - self.state
        
        # Kalman Gain calculation
        S = self.state_cov + self.measurement_cov  # Innovation covariance
        K = self.state_cov @ np.linalg.inv(S)  # Kalman Gain
        
        # Update the state vector using the measurement residual
        self.state = self.state + K @ residual
        
        # Update the covariance matrix
        self.state_cov = (self.identity - K) @ self.state_cov
    
    def get_pose(self):
        """
        Returns the current pose: quaternion (orientation), position, and velocity.

        Returns:
            Tuple: (quaternion, position, velocity)
        """
        quat = self.state[0:4]  # [qx, qy, qz, qw]
        position = self.state[4:7]  # [px, py, pz]
        velocity = self.state[7:10]  # [vx, vy, vz]
        
        return quat, position, velocity
