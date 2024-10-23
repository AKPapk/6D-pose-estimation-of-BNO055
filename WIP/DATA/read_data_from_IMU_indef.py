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
#serialCom.close()

#kmax = 500 # number of data points to read
try:
    while True:
        try:
            s_bytes = serialCom.readline()
            
            decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')
            #print(decoded_bytes)
            
            values = [float(x) for x in decoded_bytes.split(",")]
            print(values)
            #time.sleep(15)
            
            writer = csv.writer(f, delimiter=",")
            writer.writerow(values)
        
        except ValueError:
            pass
            #print("ERROR. Line was not recorded.")
except KeyboardInterrupt:
    print("\nSaving data...\n")
  
finally: 
    f.close()
    serialCom.close()