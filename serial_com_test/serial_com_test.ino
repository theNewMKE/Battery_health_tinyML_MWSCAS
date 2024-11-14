#define BUFFER_SIZE 160
// create a buffer array
// int data_buffer[BUFFER_SIZE];
float data_buffer[BUFFER_SIZE];
// set the flag to false
bool data_rec_flag = false;


void setup() {
  // put your setup code here, to run once:
  // pinMode(LED_BUILTIN, OUTPUT);

  // set baud rate
  Serial.begin(9600);
}

void loop() 
{
  //delay(7000);

  // int received_byte = 0;
  float received_byte = 0;
  int i = 0;

  // set flag to false
  data_rec_flag = false;

  // if serial is available and i less than BUFFER_SIZE
  while (Serial.available() && i < BUFFER_SIZE) {
    data_rec_flag = true;

    // read byte from serial buffer
    //  received_byte = Serial.read();
    // parse float from buffer
    received_byte = Serial.parseFloat();

    // store the byte to data_biffer array
    data_buffer[i] = received_byte;
    i++;
  }
  // print buffer;
  // if flag equls true
  if (data_rec_flag == true) {
    Serial.println("Received " + String(i) + " numbers->");
    // Serial.print("Received numbers are:\n");
    // print the batch
    for (int j=0; j<i; j++) {
      // print each element in the buffer
      Serial.print( data_buffer[j]);
      //Serial.print(atoi(data_buffer[i]));
      Serial.print(" ");
    }

    Serial.print("\n");
    delay(100);
  }


} // end of loop


// led example
  // char RxedByte = 0;
  // char data[5];
  // if (Serial.available()) 
  // {
    
  //   RxedByte = Serial.read();   
  //   // RxedByte = Serial.readBytes(data, 5);
  //   Serial.println((int)RxedByte);
  //   switch(RxedByte)
  //   {
  //     case 'A':  digitalWrite(12,HIGH);
  //       Serial.println(RxedByte);
  //       digitalWrite(LED_BUILTIN, HIGH);  // turn the LED on (HIGH is the voltage level)
  //       delay(1000);                      
  //       digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  //       delay(1000); 
  //       digitalWrite(LED_BUILTIN, HIGH);
  //       delay(1000);
  //       digitalWrite(LED_BUILTIN, LOW);   // turn the LED off by making the voltage LOW
  //       delay(1000); 
  //     case 'B': //your code
  //         break;
  //   default:
  //         break;
  //   }//end of switch()
  // }//endof if

