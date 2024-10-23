import numpy as np 

'''
ACC DATA
'''

# offsets of each variable in the state vector
iAx = 0
iAy = 1
iAz = 2
NUMVARS = iAz + 1        # number of variables in kalman filter state vector

# to have more variables, add i*varname* = next number (ex.: iHeight = 2) [[# refers to position]]
# and replave iV with last added variable in NUMVARS

class KF:
    def __init__(self, initial_ax: float, 
                       initial_ay: float,
                       initial_az: float,
                       accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)
        
        self._x[iAx] = initial_ax
        self._x[iAy] = initial_ay
        self._x[iAz] = initial_az
        
        self._accel_varaiance = accel_variance
        
        # covariance of state GRV 
        self._P = np.eye(NUMVARS)         # NEEDS TO BE MATCHED TO SPECIFICATION ON DATA SHEET [not I matrix]
        
    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)     # Since acceleration is constant, F is the indentity matrix
        
        
        G = np.eye(NUMVARS) * dt
        
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_varaiance  # update COV with process noise
        
        
        self._x = new_x
        self._P = new_P
     
        
    
    def update(self, acc_meas_value: np.array, acc_meas_variance: np.array):
         # y = z - Hx (Measurement residual)
        # S = H P H.T + R (Residual covariance)
        # K = P H.T S^-1 (Kalman gain)
        # x = x + K y (State update)
        # P = (I - K H) P (Covariance update)
        
        H = np.eye(NUMVARS)
        
        z = acc_meas_value
        R = np.diag(acc_meas_variance)

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
    def accel(self) -> np.array:
        return self._x