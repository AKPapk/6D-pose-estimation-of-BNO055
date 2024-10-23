import numpy as np


'''
TURNING OFF GRAVITY
'''


class GOFF:
    def __init__(self, g_world = np.array([0,0,0,9.81])) -> None:
        
        self.g_world = g_world

    def quaternion_mult(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quat_conjugate(self, q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def acc_linear_sub(self, a1, g1):
        x1, y1, z1 = a1
        x2, y2, z2 = g1
        return np.array([x1-x2, y1-y2, z1- z2])
    
    def turn_off_gravity(self, quat_mus_in_array, acc_mus_in_array):
        acc_linear = []
    
        for i in range(len(quat_mus_in_array)):
            q = quat_mus_in_array[i]
            q_conjugate = self.quat_conjugate(q)
            
            temp_result = self.quaternion_mult(q_conjugate, self.g_world)
            g_imu = self.quaternion_mult(temp_result, q)
            g_imu_vector = g_imu[1:]
            
            a = acc_mus_in_array[i]
            a_linear = self.acc_linear_sub(a, g_imu_vector)
            
            acc_linear.append(a_linear)
            
        return np.array(acc_linear)
        
        
    
    
