#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

#define BNO055_SAMPLERATE_DELAY_MS (100)

Adafruit_BNO055 myIMU = Adafruit_BNO055();

void setup() {
  Serial.begin(115200);
  if (!myIMU.begin()) {
    Serial.print("No BNO055 detected");
    while(1);
  }
  myIMU.setExtCrystalUse(true);
  //myIMU.setMode(Adafruit_BNO055::OPERATION_MODE_IMU)
  //int8_t temp = myIMU.getTemp();
  //Serial.println(temp);

/*
  //================ CALLIBRATION ==================//
  uint8_t system, gyro,  accel, mg = 0;
  while (system < 3 || gyro < 3 || accel < 3 || mg < 3) {
    bno.getCalibration(&system, &gyro, &accel, &mg);
    Serial.print(accel);
    Serial.print(", ");
    Serial.print(gyro);
    Serial.print(", ");
    Serial.print(mg);
    Serial.print(", ");
    Serial.print(system);
    Serial.println("\t");
  }
  Serial.print("CALIBRATED");
*/
  //delay(100);
  Serial.print("acc_x, ");
  Serial.print("acc_y, ");
  Serial.print("acc_z");
  Serial.print("quat_w");
  Serial.print("quat_x, ");
  Serial.print("quat_y, ");
  Serial.print("quat_z");
  Serial.println();
  delay(100);
}

void loop() {
  imu::Vector<3> acc = myIMU.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> gyr = myIMU.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
  imu::Vector<3> mag = myIMU.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
  imu::Quaternion quat = myIMU.getQuat();


  Serial.print(acc.x());
  Serial.print(" ");
  Serial.print(acc.y());
  Serial.print(" ");
  Serial.print(acc.z());
  Serial.print(" ");


  Serial.print(quat.w());
  Serial.print(" ");
  Serial.print(quat.x());
  Serial.print(" ");
  Serial.print(quat.y());
  Serial.print(" ");
  Serial.println(quat.z());
/*
  Serial.print("||");

  Serial.print(gyr.x());
  Serial.print(",");
  Serial.print(gyr.y());
  Serial.print(",");
  Serial.print(gyr.z());

  Serial.print("||");
  
  Serial.print(mag.x());
  Serial.print(",");
  Serial.print(mag.y());
  Serial.print(",");
  Serial.println(mag.z());
  */

  //delay(BNO055_SAMPLERATE_DELAY_MS);
}
