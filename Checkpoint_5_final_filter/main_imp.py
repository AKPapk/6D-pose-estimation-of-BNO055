from kf_imp import KF
from Quat_kf import QKF
from Gravity_OFF import GOFF
from integration import RK4Integrator
from fusion_kf import PKF
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

#integrate_acc = MATHS(dt=DT)


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

#velocity, position = integrate_acc.integrate_rk4(acc_linear)
rk4_integrator = RK4Integrator(dt=DT)

# Perform integration to get velocity and position
velocity, position = rk4_integrator.integrate(acc_linear)

# Call kalman filter to fuse orientation and position data
pose_kf = PKF(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)

pose_covs = []
pose_mus = []

#pose_kf.update_pose(position, velocity, quat_mus_in_array, pose_variance=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1, 0.1, 0.1]))
for step in range(NUM_STEPS):
    
    pose_covs.append(pose_kf.cov)
    pose_mus.append(pose_kf.mean)
    
    position_data = position[step]
    velocity_data = velocity[step]
    quat_data = quat_mus_in_array[step]
    #pose_variance_data = pose_variance[step]
    
    pose_kf.predict_pose(dt=DT)

    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        pose_measurement = np.hstack([position_data, velocity_data, quat_data])
        pose_kf.update_pose(pose_measurement=pose_measurement, pose_variance=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))




pos_fused = pose_mus[0:3]
vel_fused = pose_mus[3:6]
quat_fused = pose_mus[6:10]



plt.subplot(4, 4, 1)
plt.title('Acceleration X')
plt.plot([acc_mu[0] for acc_mu in acc_mus], 'r', label = 'Filtered ax')
plt.plot(ax_data, 'b', label = 'Real ax')
plt.plot([acc_mu_in_array[0]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[0] - 2*np.sqrt(cov[0,0]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')         # bounds to show uncretainity if there are no new measurements
plt.plot([acc_mu[0] + 2*np.sqrt(cov[0,0]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()

plt.subplot(4, 4, 5)
plt.title('Acceleration Y')
plt.plot([acc_mu[1] for acc_mu in acc_mus], 'r', label = 'Filtered ay')
plt.plot(ay_data, 'b', label = 'Real ay')
plt.plot([acc_mu_in_array[1]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[1] - 2*np.sqrt(cov[1,1]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')
plt.plot([acc_mu[1] + 2*np.sqrt(cov[1,1]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()

plt.subplot(4, 4, 9)
plt.title('Acceleration Z')
plt.plot([acc_mu[2] for acc_mu in acc_mus], 'r', label = 'Filtered az')
plt.plot(az_data, 'b', label = 'Real az')
plt.plot([acc_mu_in_array[2]for acc_mu_in_array in acc_linear], 'g', label = 'W/o gravity')
plt.plot([acc_mu[2] - 2*np.sqrt(cov[2,2]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--', label = 'Uncertainty')
plt.plot([acc_mu[2] + 2*np.sqrt(cov[2,2]) for acc_mu, cov in zip(acc_mus,acc_covs)], 'r--')
plt.legend()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

plt.subplot(4, 4, 2)
plt.title('Quaternion W')
plt.plot([quat_mu[0] for quat_mu in quat_mus], 'r', label = 'Filtered qw')
plt.plot(qw_data, 'b', label = 'Real qw')
plt.plot([vel[6]for vel in pose_mus], 'g', label = 'Filtered 2 qw')
plt.plot([quat_mu[0] - 2*np.sqrt(cov[0,0]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')         # bounds to show uncretainity if there are no new measurements
plt.plot([quat_mu[0] + 2*np.sqrt(cov[0,0]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 4, 6)
plt.title('Quaternion X')
plt.plot([quat_mu[1] for quat_mu in quat_mus], 'r', label = 'Filtered qx')
plt.plot(qx_data, 'b', label = 'Real qx')
plt.plot([vel[7]for vel in pose_mus], 'g', label = 'Filtered 2 qx')
plt.plot([quat_mu[1] - 2*np.sqrt(cov[1,1]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[1] + 2*np.sqrt(cov[1,1]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 4, 10)
plt.title('Quaternion Y')
plt.plot([quat_mu[2] for quat_mu in quat_mus], 'r', label = 'Filtered qy')
plt.plot(qy_data, 'b', label = 'Real qy')
plt.plot([vel[8]for vel in pose_mus], 'g', label = 'Filtered 2 qy')
plt.plot([quat_mu[2] - 2*np.sqrt(cov[2,2]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[2] + 2*np.sqrt(cov[2,2]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

plt.subplot(4, 4, 14)
plt.title('Quaternion Z')
plt.plot([quat_mu[3] for quat_mu in quat_mus], 'r', label = 'Filtered qz')
plt.plot(qz_data, 'b', label = 'Real qz')
plt.plot([vel[9]for vel in pose_mus], 'g', label = 'Filtered 2 qy')
plt.plot([quat_mu[3] - 2*np.sqrt(cov[3,3]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--', label = 'Uncertainty')
plt.plot([quat_mu[3] + 2*np.sqrt(cov[3,3]) for quat_mu, cov in zip(quat_mus,quat_covs)], 'r--')
plt.legend()

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


plt.subplot(4, 4, 3)
plt.title('Velocity X')
plt.plot([vel[0]for vel in velocity], 'b', label = 'Velocity X')
plt.plot([vel[3]for vel in pose_mus], 'r', label = 'Velocity Filtered X')
plt.legend()

plt.subplot(4, 4, 7)
plt.title('Velocity Y')
plt.plot([vel[1]for vel in velocity], 'b', label = 'Velocity Y')
plt.plot([vel[4]for vel in pose_mus], 'r', label = 'Velocity Filtered Y')
plt.legend()

plt.subplot(4, 4, 11)
plt.title('Velocity Z')
plt.plot([vel[2]for vel in velocity], 'b', label = 'Velocity Z')
plt.plot([vel[5]for vel in pose_mus], 'r', label = 'Velocity Filtered Z')
plt.legend()
#this graph shows how when the measured value gets updated, the estimate becomes more and more accurate as time goes on

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

plt.subplot(4, 4, 4)
plt.title('Position X')
plt.plot([pos[0] for pos in position], 'b', label = 'Position X')
plt.plot([pos[0] for pos in pose_mus], 'r', label = 'Position Filtered X')
plt.legend()

plt.subplot(4, 4, 8)
plt.title('Position Y')
plt.plot([pos[1] for pos in position], 'b', label = 'Position Y')
plt.plot([pos[1] for pos in pose_mus], 'r', label = 'Position Filtered Y')
plt.legend()

plt.subplot(4, 4, 12)
plt.title('Position Z')
plt.plot([pos[2] for pos in position], 'b', label = 'Position Z')
plt.plot([pos[2] for pos in pose_mus], 'r', label = 'Position Filtered Z')
plt.legend()


plt.show()
#plt.ginput(2)