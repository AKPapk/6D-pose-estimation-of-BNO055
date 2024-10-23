from sqlite3 import TimestampFromTicks
from kf_imp import KF
from Quat_kf import QKF
from Gravity_OFF import GOFF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

file_dir = './MIGHT_FUCK_UP/DATA/wQUATdata.csv'


ax_data, ay_data, az_data = [], [], []
qw_data, qx_data, qy_data, qz_data = [], [], [], []

with open(file_dir, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        ax_data.append(float(row[0]))
        ay_data.append(float(row[1]))
        az_data.append(float(row[2]))
        
        qw_data.append(float(row[3]))
        qx_data.append(float(row[4]))
        qy_data.append(float(row[5]))
        qz_data.append(float(row[6]))
        
ax_data = np.array(ax_data)
ay_data = np.array(ay_data)
az_data = np.array(az_data)

qw_data = np.array(qw_data)
qx_data = np.array(qx_data)
qy_data = np.array(qy_data)
qz_data = np.array(qz_data)


DT = 0.1        # uppercase bc it's a constant
NUM_STEPS = len(ax_data)
MEAS_EVERY_STEPS = 1

acc_kf = KF(initial_ax=0.0, initial_ay=0.0, initial_az=0.0, accel_variance=0.1) # how TF is variance calculated

quat_kf = QKF(initial_qw=0.0, initial_qx=0.0, initial_qy=0.0, initial_qz=0., quat_variance=0.1)

acc_goff = GOFF()


acc_mus = []
quat_mus = []

acc_covs = []
quat_covs = []

real_accels = {"ax": ax_data, "ay": ay_data, "az": az_data}
real_quats = {"qw": qw_data, "qx": qx_data, "qy": qy_data, "qz": qz_data}


for step in range(NUM_STEPS):
        
        
    acc_covs.append(acc_kf.cov)
    quat_covs.append(quat_kf.cov)
    
    acc_mus.append(acc_kf.mean)
    quat_mus.append(quat_kf.mean)
    
    real_ax = ax_data[step]
    real_ay = ay_data[step]
    real_az = az_data[step]
    
    real_qw = qw_data[step]
    real_qx = qx_data[step]
    real_qy = qy_data[step]
    real_qz = qz_data[step]
    
    acc_kf.predict(dt=DT)
    quat_kf.predict_orient(dt=DT)
    
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        acc_measurement = np.array([real_ax, real_ay, real_az])
        acc_kf.update(acc_meas_value=acc_measurement, acc_meas_variance=np.array([0.01, 0.01, 0.01]))      # measuremnt noise
        
        quat_measurement = np.array([real_qw, real_qx, real_qy, real_qz])
        quat_kf.update_orient(quat_meas_value=quat_measurement, quat_meas_variance=np.array([0.01, 0.01, 0.1, 0.1]))


quat_mus_in_array = np.array(quat_mus)
acc_mus_in_array = np.array(acc_mus)    


acc_linear = acc_goff.turn_off_gravity(quat_mus_in_array, acc_mus_in_array)


plt.subplot(4, 2, 1)
plt.title('Acceleration X')
plt.plot([acc_mu[0] for acc_mu in acc_mus], 'r', label = 'Filtered ax')
plt.plot(ax_data, 'b', label = 'Real ax')
plt.plot([acc_mu_in_array[0]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[0] - 2*np.sqrt(cov[0,0]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')         # bounds to show uncretainity if there are no new measurements
plt.plot([acc_mu[0] + 2*np.sqrt(cov[0,0]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()

plt.subplot(4, 2, 2)
plt.title('Acceleration Y')
plt.plot([acc_mu[1] for acc_mu in acc_mus], 'r', label = 'Filtered ay')
plt.plot(ay_data, 'b', label = 'Real ay')
plt.plot([acc_mu_in_array[1]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[1] - 2*np.sqrt(cov[1,1]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')
plt.plot([acc_mu[1] + 2*np.sqrt(cov[1,1]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()

plt.subplot(4, 2, 3)
plt.title('Acceleration Z')
plt.plot([acc_mu[2] for acc_mu in acc_mus], 'r', label = 'Filtered az')
plt.plot(az_data, 'b', label = 'Real az')
plt.plot([acc_mu_in_array[2]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[2] - 2*np.sqrt(cov[2,2]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')
plt.plot([acc_mu[2] + 2*np.sqrt(cov[2,2]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
plt.subplot(4, 2, 4)
plt.title('Quaternion W')
plt.plot([quat_mu[0] for quat_mu in quat_mus], 'r', label = 'Filtered qw')
plt.plot(qw_data, 'b', label = 'Real qw')
plt.plot([quat_mu[0] - 2*np.sqrt(cov[0,0]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')         # bounds to show uncretainity if there are no new measurements
plt.plot([quat_mu[0] + 2*np.sqrt(cov[0,0]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 2, 5)
plt.title('Quaternion X')
plt.plot([quat_mu[1] for quat_mu in quat_mus], 'r', label = 'Filtered qx')
plt.plot(qx_data, 'b', label = 'Real aqx')
plt.plot([quat_mu[1] - 2*np.sqrt(cov[1,1]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[1] + 2*np.sqrt(cov[1,1]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 2, 6)
plt.title('Quaternion Y')
plt.plot([quat_mu[2] for quat_mu in quat_mus], 'r', label = 'Filtered qy')
plt.plot(qy_data, 'b', label = 'Real qy')
plt.plot([quat_mu[2] - 2*np.sqrt(cov[2,2]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[2] + 2*np.sqrt(cov[2,2]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 2, 7)
plt.title('Quaternion Z')
plt.plot([quat_mu[3] for quat_mu in quat_mus], 'r', label = 'Filtered qz')
plt.plot(qz_data, 'b', label = 'Real qz')
plt.plot([quat_mu[3] - 2*np.sqrt(cov[3,3]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[3] + 2*np.sqrt(cov[3,3]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()


#this graph shows how when the measured value gets updated, the estimate becomes more and more accurate as time goes on

plt.tight_layout()
plt.show()
#plt.ginput(2)