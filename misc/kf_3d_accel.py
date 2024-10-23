
import numpy as np

# Offsets of each variable in the state vector (ax, ay, az)
iAx = 0
iAy = 1
iAz = 2
NUMVARS = iAz + 1  # Number of variables in Kalman filter state vector

class KF_Accel3D:
    def __init__(self, initial_ax: float, initial_ay: float, initial_az: float, accel_variance: float) -> None:
        # Mean of state (ax, ay, az)
        self._x = np.zeros(NUMVARS)
        
        self._x[iAx] = initial_ax
        self._x[iAy] = initial_ay
        self._x[iAz] = initial_az
        
        self._accel_variance = accel_variance
        
        # Covariance of state GRV 
        self._P = np.eye(NUMVARS)  # Identity matrix as initial covariance matrix (uncertainty)

    def predict(self, dt: float) -> None:
        # x = F x (State prediction)
        # P = F P F.T + Q (Covariance prediction)
        F = np.eye(NUMVARS)  # Since acceleration is constant, F is the identity matrix
        
        # G matrix for process noise addition (based on acceleration)
        G = np.eye(NUMVARS) * dt  # G should scale process noise by time
        
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance  # Update covariance with process noise
        
        self._x = new_x
        self._P = new_P

    def update(self, meas_value: np.array, meas_variance: np.array):
        # y = z - Hx (Measurement residual)
        # S = H P H.T + R (Residual covariance)
        # K = P H.T S^-1 (Kalman gain)
        # x = x + K y (State update)
        # P = (I - K H) P (Covariance update)
        
        H = np.eye(NUMVARS)  # Measurement matrix, assuming direct measurement of acceleration
        
        z = meas_value  # Measurement values for ax, ay, az
        R = np.diag(meas_variance)  # Measurement noise covariance (variance for each axis)

        y = z - H.dot(self._x)  # Innovation (difference between measurement and prediction)
        S = H.dot(self._P).dot(H.T) + R  # Innovation covariance
        K = self._P.dot(H.T).dot(np.linalg.inv(S))  # Kalman gain

        self._x = self._x + K.dot(y)  # Update state estimate
        self._P = (np.eye(NUMVARS) - K.dot(H)).dot(self._P)  # Update covariance matrix

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def accel(self) -> np.array:
        return self._x
