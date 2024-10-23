from statistics import variance
import numpy as np

'''
FUSION FOR POSE
'''

iPx = 0
iPy = 1
iPz = 2
iVx = 3
iVy = 4
iVz = 5
iQw = 6
iQx = 7
iQy = 8
iQz = 9
NUMVARS = iQz + 1

class PKF:
    def __init__(self, initial_px: float,
                       initial_py: float,
                       initial_pz: float,
                       initial_vx: float,
                       initial_vy: float,
                       initial_vz: float,
                       initial_qw: float,
                       initial_qx: float,
                       initial_qy: float,
                       initial_qz: float,
                       variance_place_holder: float) -> None:
      
        self._x = np.zeros(NUMVARS)

        self._x[iPx] = initial_px
        self._x[iPy] = initial_py
        self._x[iPz] = initial_pz
        self._x[iVx] = initial_vx
        self._x[iVy] = initial_vy
        self._x[iVz] = initial_vz
        self._x[iQw] = initial_qw
        self._x[iQx] = initial_qx
        self._x[iQy] = initial_qy
        self._x[iQz] = initial_qz
        
        self._variance = variance_place_holder
        
        self._P = np.eye(NUMVARS)

    def predict_pose(self, dt: float) -> None:
      
        F= np.eye(NUMVARS)
        G = np.eye(NUMVARS) * dt
        new_x = F.dot(self._x)
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._variance  # update COV with process noise
        
        self._x = new_x
        self._P = new_P
        
        
    ## position: np.array, velocity: np.array, quat_mus_in_array: np.array,
    def update_pose(self, pose_measurement: np.array,  pose_variance: np.array):
    
        #measurements = np.hstack((velocity, position, quat_mus_in_array))
        
        H = np.eye(NUMVARS)
        
        z = pose_measurement
        R = np.diag(pose_variance)

        y = z - H.dot(self._x)
        
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))        # might be better to have sudo inverse instead of inverse
        
        #print(velocity.shape)
        #print(y.shape)
        
        new_x = self._x + K.dot(y)
        new_P = (np.eye(NUMVARS) - K.dot(H)).dot(self._P)
        
        self._P = new_P 
        self._x = new_x
        
    def get_pose(self):
        
        pos_fused = self._x[0:3]    # [vx, vy, vz]
        vel_fused = self._x[3:6]    # [px, py, pz]
        quat_fused = self._x[6:10]  # [qx, qy, qz, qw]
        
        return pos_fused, vel_fused, quat_fused
        
        
    @property
    def cov(self) -> np.array:
        return self._P
    
    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pose(self) -> np.array:
        return self._x