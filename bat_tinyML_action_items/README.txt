Folders and files:
1. cc_file has 4 .cc g_models-> quant and no_quant models for tar1 and tar2, 
you will most likely not use any of this, however you cold drage the file into
Netron if you want to.

2. hello_world_V2 has the whole project downloaded from dejazzer website,
this is the regression example, which is what we need.
I have commented out the original g_model it has and copy pasted the
quantized model for tar1 in model.cpp file in hello_world_V2 folder.

3. original_model_sum has the original models summary for tar1 and tar2,
since you asked me today so I copy and pasted from the model.summary() 
for the original models.

4. pickled_data has X_test, y1_test, and y2_test, which you will 
need to transfer X_test to the board by the python code and get y1_predict 
from the board (g_model) and then compare y1_predict to y1_test.

5. serial_com_test (foler) has the .ino file that we worked on last time for 
serial transfer data, I changed a little so that it can take float numbers now.
The code has comments.

6. serial_com_test.py is the code that we used pyserial for the serial communication.
It also have comments. 


Please first look at serial_com_test.py and serial_com_test.ino, so that you could 
know how to transfer float data and what kind of data we are using for our model 
(input size and shape). 
Then work on the hello_world_V2 example to see if we could get y1_predict out of it.

I checked again (on that slide), the Ardiuno Nano 33 BLE has 1 MB flash memory,
and 256 KB SRAM. Our models (tflite quantized) have 62 and 74 KB on flash, 
but not sure how much they will take on RAM. 

Please do not forget to investigate the inference time as well if possible.

To be clear, I want to emphasis the shape of the input (X_test),
which is (3560, 20, 8). This means we feed into the g_model (20,8) each time and 
we loop this for 3560 times, so that we can get 3560 prediction values, 
then transfer it back to computer. 
This information is also in serial_com_test.py


