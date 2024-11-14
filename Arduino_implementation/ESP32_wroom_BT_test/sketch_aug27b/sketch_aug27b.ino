// // #include "BluetoothSerial.h"

// // BluetoothSerial serialBT;
// // char cmd;

// // void setup() {
// //   // put your setup code here, to run once:
// //   serialBT.begin("ESP32-BT");
// //   pinMode(2, OUTPUT);
// // }

// // void loop() {
// //   // put your main code here, to run repeatedly:
// //   if(serialBT.available()){
// //       cmd = serialBT.read();
// //   }
// //   if(cmd == '1'){
// //     digitalWrite(2, HIGH);
// //   }
// //   if(cmd == '0'){
// //     digitalWrite(2, LOW);
// //   }
// //   delay(20);
// // }


// #include "BluetoothSerial.h"

// // Create a BluetoothSerial object
// BluetoothSerial SerialBT;

// void setup() {
//   // Start the serial communication
//   Serial.begin(9600);

//   // Start the Bluetooth serial with a name for the device
//   SerialBT.begin("ESP32_BT_Device"); 
//   Serial.println("Bluetooth device is ready to pair");

//   // Wait for a connection
//   int n = 1;
//   while (!SerialBT.hasClient()) {
//     delay(3000);
//     Serial.println("Waiting for Bluetooth connection...");
//     Serial.print(n);
//     n++;
//   }
//   Serial.println("Bluetooth connected!");
// }

// void loop() {
//   // // Example: Automatically send data every second
//   // String dataToSend = "Hello from ESP32!";  // Replace with your data source
//   // SerialBT.println(dataToSend);             // Send data via Bluetooth
//   // Serial.println("Data sent: " + dataToSend);  // Log the sent data
//   // delay(1000);  // Wait 1 second before sending the next message
//     // Check if data is available from Bluetooth
//   if (SerialBT.available()) {
//     String receivedData = SerialBT.readStringUntil('\n');  // Read the incoming data
//     Serial.println("Received from PC: " + receivedData);  // Print the received data

//     // Optionally, you can process the data
//     // Example: Echo the data back to the PC
//     SerialBT.println("Echo: " + receivedData);
//   }

//   delay(100);  // Small delay to avoid overwhelming the loop
// }



/*
  Bluetooth Serial Client Demo
  bluetooth_serial_client.ino
  Demonstrates operation of BluetoothSerial Library
  Based upon open-source code sample by Evandro Copercini - 2018
  
  DroneBot Workshop 2024
  https://dronebotworkshop.com
*/
 
// Include BluetoothSerial library.
#include "BluetoothSerial.h"
 
// Name of Bluetooth client.
String device_name = "ESP32-BT-Client";
 
// Check if Bluetooth is available.
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled!
#endif
 
// Check if Serial Port Profile (SPP) is available.
#if !defined(CONFIG_BT_SPP_ENABLED)
#error Serial Port Profile for Bluetooth is not available or not enabled!
#endif
 
// Create a BluetoothSerial object.
BluetoothSerial SerialBT;
 
void setup() {
  // Start Serial Monitor.
  Serial.begin(9600);
 
  // Start Bluetooth Serial.
  SerialBT.begin(device_name);
 
  // Uncomment the next line to delete previously paired devices. Must be called directly after bluetooth begin.
  //SerialBT.deleteAllBondedDevices();
 
  // Print to serial monitor.
  Serial.printf("The device \"%s\" is started and can be paired with Bluetooth.\n", device_name.c_str());
}
 
void loop() {
  // Check for messsage from serial monitor.
  if (Serial.available()) {
    // Write message to paired Bluetooth device.
    SerialBT.write(Serial.read());
  }
 
  // Check for message from paired Bluetooth device.
  if (SerialBT.available()) {
    // Write message to serial monitor.
    Serial.write(SerialBT.read());
  }
  delay(20);
}

