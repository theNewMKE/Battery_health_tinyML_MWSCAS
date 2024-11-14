#define BUFFER_SIZE 160
float data_buffer[BUFFER_SIZE];
bool data_rec_flag = false;


void setup() 
{
  Serial.begin(9600);
  while(!Serial);
  delay(100);
}

void loop() 
{
  // int received_byte = 0;
  float received_byte = 0;
  int i = 0;
  // set flag to false
  data_rec_flag = false;

  while (Serial.available() && i < BUFFER_SIZE) {
    data_rec_flag = true;
    received_byte = Serial.parseFloat();
    // store the byte to data_buffer array
    data_buffer[i] = received_byte;
    i++;
  }

  if (data_rec_flag == true) {
    Serial.println(i); // "\r\n"
    /*---
    Serial.println("Received " + String(i) + " numbers:"); // "\r\n"
    for (int j=0; j<i; j++) {
      Serial.print(data_buffer[j]);
      if (j < i-1) { Serial.print(" "); }
    }
    Serial.print("\r\n"); // "\r\n"
    ---*/
    delay(100);
  }

} 
