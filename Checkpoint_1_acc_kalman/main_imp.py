from kf_imp import KF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

file_dir = './MIGHT_FUCK_UP/DATA/testDATA1.csv'

#### = pandas

#### data = pd.read_csv(file_dir)

#### ax_data = data['ax'].values
#### ay_data = data['ay'].values
#### az_data = data['az'].values
ax_data, ay_data, az_data = [], [], []

with open(file_dir, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        ax_data.append(float(row[0]))
        ay_data.append(float(row[1]))
        az_data.append(float(row[2]))
        
ax_data = np.array(ax_data)
ay_data = np.array(ay_data)
az_data = np.array(az_data)

#plt.ion()
#plt.figure()

#real_Ax, real_Ay, real_Az = 0.0, 0.0, 0.0
#meas_variance = np.array([0.1 ** 2, 0.1 ** 2, 0.1 **2])
#real_Vx, real_Vy, real_Vz = 0.9, 0.5, 0.3

DT = 0.1        # uppercase bc it's a constant
NUM_STEPS = len(ax_data)
MEAS_EVERY_STEPS = 1

kf = KF(initial_ax=0.0, initial_ay=0.0, initial_az=0.0, accel_variance=0.1)



mus = [] 
covs = []
real_accels = {"ax": ax_data, "ay": ay_data, "az": az_data}


for step in range(NUM_STEPS):
    
    #if step > 500:
     #   real_Vx *= 0.9
    #    real_Vy *= 0.95
     #   real_Vz *= 0.92
        
        
    covs.append(kf.cov)
    mus.append(kf.mean)
    
    real_ax = ax_data[step]
    real_ay = ay_data[step]
    real_az = az_data[step]
    
    kf.predict(dt=DT)
    
    if step != 0 and step % MEAS_EVERY_STEPS == 0:
        measurement = np.array([real_ax, real_ay, real_az])
        kf.update(meas_value= measurement, meas_variance=np.array([0.01, 0.01, 0.01]))      # measurement noise
    
    #real_accels["Ax"].append(real_Ax)
    #real_accels["Ay"].append(real_Ay)
    #real_accels["Az"].append(real_Az)

        
        


plt.subplot(3, 1, 1)
plt.title('Acceleration X')
plt.plot([mu[0] for mu in mus], 'r', label = 'Filtered ax')
plt.plot(ax_data, 'b', label = 'Real ax')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--', label = 'Uncertainty')         # bounds to show uncretainity if there are no new measurements
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus,covs)], 'r--')
plt.legend()

plt.subplot(3, 1, 2)
plt.title('Acceleration Y')
plt.plot([mu[1] for mu in mus], 'r', label = 'Filtered ay')
plt.plot(ay_data, 'b', label = 'Real ay')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--', label = 'Uncertainty')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus,covs)], 'r--')
plt.legend()

plt.subplot(3, 1, 3)
plt.title('Acceleration Z')
plt.plot([mu[2] for mu in mus], 'r', label = 'Filtered az')
plt.plot(az_data, 'b', label = 'Real az')
plt.plot([mu[2] - 2*np.sqrt(cov[2,2]) for mu, cov in zip(mus,covs)], 'r--', label = 'Uncertainty')
plt.plot([mu[2] + 2*np.sqrt(cov[2,2]) for mu, cov in zip(mus,covs)], 'r--')
plt.legend()

#this graph shows how when the measured value gets updated, the estimate becomes more and more accurate as time goes on

plt.tight_layout()
plt.show()
#plt.ginput(2)