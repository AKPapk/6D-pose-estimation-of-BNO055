import numpy as np 

'''
QUAT DATA
'''

# offsets of each variable in the state vector
iQw = 0
iQx = 1
iQy = 2
iQz = 3
NUMVARS = iQz + 1        # number of variables in kalman filter state vector

# to have more variables, add i*varname* = next number (ex.: iHeight = 2) [[# refers to position]]
# and replave iV with last added variable in NUMVARS

class QKF:
    def __init__(self, initial_qw: float,
                       initial_qx: float,
                       initial_qy: float,
                       initial_qz: float,
                       quat_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)
        
        self._x[iQw] = initial_qw
        self._x[iQx] = initial_qx
        self._x[iQy] = initial_qy
        self._x[iQz] = initial_qz
        
        self._quat_varaiance = quat_variance
        
        # covariance of state GRV 
        self._P = np.eye(NUMVARS)         # NEEDS TO BE MATCHED TO SPECIFICATION ON DATA SHEET [not I matrix]
        
    def predict_orient(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)     # Since acceleration is constant, F is the indentity matrix
        
        
        G = np.eye(NUMVARS) * dt
        
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._quat_varaiance  # update COV with process noise
    
        
        
        self._x = new_x
        self._P = new_P
     
        
    
    def update_orient(self, quat_meas_value: np.array, quat_meas_variance: np.array):
         # y = z - Hx (Measurement residual)
        # S = H P H.T + R (Residual covariance)
        # K = P H.T S^-1 (Kalman gain)
        # x = x + K y (State update)
        # P = (I - K H) P (Covariance update)
        
        H = np.eye(NUMVARS)
        
        z = quat_meas_value
        R = np.diag(quat_meas_variance)

        y = z - H.dot(self._x)
        
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))        # might be better to have sudo inverse instead of inverse
        
        new_x = self._x + K.dot(y)
        new_P = (np.eye(NUMVARS) - K.dot(H)).dot(self._P)
        
        self._P = new_P 
        self._x = new_x
        
        
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def quat(self) -> np.array:
        return self._x