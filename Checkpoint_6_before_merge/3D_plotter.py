import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import csv

import pydantic

#from misc.Main import MEAS_EVERY_STEPS

file_dir = './MIGHT_FUCK_UP/DATA/GT_but_cute.csv'


px_data, py_data, pz_data = [], [], []
qw_data, qx_data, qy_data, qz_data = [], [], [], []

with open(file_dir, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        px_data.append(float(row[4]))
        py_data.append(float(row[5]))
        pz_data.append(float(row[6]))
        
        qw_data.append(float(row[3]))
        qx_data.append(float(row[0]))
        qy_data.append(float(row[1]))
        qz_data.append(float(row[2]))
        
px_data = np.array(px_data)
py_data = np.array(py_data)
pz_data = np.array(pz_data)


qw_data = np.array(qw_data)
qx_data = np.array(qx_data)
qy_data = np.array(qy_data)
qz_data = np.array(qz_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_box_aspect([1, 1, 0.01])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.plot(px_data*100, pz_data*100, py_data, c='b', label="Position")

plt.title('IMU 3D Position and Orientation')
plt.legend()
plt.show()