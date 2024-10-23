import serial
import time
import csv

#ports = list_ports.comports()
#for port in ports: print(port)

f = open("TEST1_w_GT.csv", "w", newline='') 
f.truncate()

serialCom = serial.Serial('COM3', 115200)
keyword = "CALIBRATED"
labels = ["Acc", "Gyro", "Mag", "Sys"]



# Reset arduino after connecting to get most relevant data
serialCom.setDTR(False)
time.sleep(1)
serialCom.flushInput()
serialCom.setDTR(True)

kmax = 501 # number of data points to read
for k in range(kmax):     # +1 bc first line is legend
    try:
        s_bytes = serialCom.readline()
        
        decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')
        #print(decoded_bytes)
        
        if k == 0:
            values = decoded_bytes.split(",")
            #time.sleep(15)
        else:
            values = [float(x) for x in decoded_bytes.split()]
        print(values)

        writer = csv.writer(f, delimiter=",")
        writer.writerow(values)
        
    except ValueError:
        print("yo mama")
        pass
        #print("ERROR. Line was not recorded.")
    
f.close()
serialCom.close()