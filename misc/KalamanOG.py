'''import serial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Serial Connection Setup
ser = serial.Serial('com4', 115200)         # CHANGE TO SERIAL PORT IN USE

'''

'''

x = [w, x, y, z]             # State Vector

Pw = 1
Px = 1
Py = 1
Pz = 1

P = [[Pw,0,0,0],              # Error Covariance Matrix
     [0,Px,0,0],
     [0,0,Py,0],
     [0,0,0,Pz]]

Qw = 1
Qx = 1
Qy = 1
Qz = 1

Q = [[Qw,0,0,0],              # Process Noise Covariance
     [0,Qx,0,0],
     [0,0,Qy,0],
     [0,0,0,Qz]]

Rw = 1
Rx = 1
Ry = 1
Rz = 1

R = [[Rw,0,0,0],              # Measurement Noise Covariance
     [0,Rx,0,0],
     [0,0,Ry,0],
     [0,0,0,Rz]]

'''

'''

x_next = *

'''