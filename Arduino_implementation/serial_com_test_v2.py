# NOTE: you may need to do: "pip install pyserial" first.
from time import sleep
import serial
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
# from utils import get_pickled_X_test_reshaped, get_pickled_y_test_reshaped
# from model_evaluation import ModelEvaluation
from utils import get_pickled_X_test_reshaped, get_pickled_y_test_reshaped
from model_evaluation import ModelEvaluation

target = 'tar1'   # tar1 tar2
model = 'TCN'   # TCN CNN

# get corresponding X and y data
X_test_reshaped = get_pickled_X_test_reshaped(target, model)
y_test_reshaped = get_pickled_y_test_reshaped(target, model)


print(f"The total input size, input type, and input dtype: "
      f"{X_test_reshaped.shape}, {type(X_test_reshaped)}, {X_test_reshaped.dtype}")
print(f"The total output size, output type, and output dtype: "
      f"{y_test_reshaped.shape}, {type(y_test_reshaped)}, {y_test_reshaped.dtype}")


print("The shape of the first input :", X_test_reshaped[0].shape)
# if you want to check the first input
#print("The first input:", X_test_reshaped[0])


# NOTE: to eliminate "\n" inserted after converting from array to string;
# read this for more details:
# https://stackoverflow.com/questions/62566893/where-is-this-n-coming-from-in-my-arrays-python
np.set_printoptions(linewidth=96)


###############################################################################
#
# Part 1: <-------------- used only for debugging purposes !!!
# next function and testing of it is to be used only with 
# Arduino sketch "arduino_serial_com_test.ino"
#
###############################################################################


# (1) send one datapoint from the dataset; that is a row with 20*8 float numbers;
# here we only transferred one datapoint, we will need to loop through 
# the whole data (X_test_reshaped) later;
# this function s to be used with the Arduino sketch "serial_com_test.ino"
# that receives 160 numbers and just echoes them back! 
def send_one_datapoint_to_arduino_1( serial_port, dp_index):
    # sleep for 2 seconds if necessary;
    # time.sleep(2)  
    # data_array is a 2D numpy array
    data_array = X_test_reshaped[dp_index]
    #print("data_array: ", data_array)
    #print("Shape of data_array: ", data_array.shape)
    #print()
    # convert the whole array to string first

    data_array = data_array.flatten()
    #print("data_array flattened: ", data_array)
    #print("Shape of data_array: ", data_array.shape) #data_array.shape
    #print()    
    
    data_array_as_string = str(data_array)
    # encode the string by using utf-8 (https://en.wikipedia.org/wiki/UTF-8)
    data_bytes = data_array_as_string.encode('utf-8')    
    # print data_array encoded
    #print("data_array encoded to bytes: ", data_bytes)
    #print("Length of sent data_array: ", len(data_bytes)) 
    #print()  
    # send to Arduino board by writing to serial
    serial_port.write(data_bytes)
    sleep(0.02)
    # read line from serial port; a line ends with "\r\n"
    # so make sure in Arduino sketch we send everything separated by 
    # say space character " " and only at the end of a transmission include
    # "\r\n", which here will mean stop reading a line; 
    #received_bytes = serial_port.readline()
    #print(received_bytes)
    received_bytes = serial_port.readline()
    decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
    print(decoded_bytes)
    #print("Length of received decoded_bytes: ", len(decoded_bytes)) 
    #print()    
    
    # converting from string to float
    # split the string
    str_splits = decoded_bytes.split()
    # to store float numbers
    str_to_float = []
    # iterate through the string
    received_count = 0
    for str_split in str_splits:
        str_split = int(str_split)  # str_split = float(str_split)
        # append the float number to the list
        received_count = received_count + 1
        str_to_float.append(str_split)   
    #print("Float number list: ", str_to_float)
    #print("Float number list length: ", received_count) #len(str_to_float)
    return str_to_float[0]
    

# # (2) test the above function; before and after the call
# # of the function, open Serial and then close Serial;
# # uncomment the next lines to do the testing:
# ser = serial.Serial("COM6", 9600) 
# send_one_datapoint_to_arduino_1( ser, 19) 
# ser.close()
# print()
# # go through all inputs and send them one by one to Arduino sketch that;
# # receive back only the number of numbers the Arduino had received; check if 
# # that is right or not;
# counts_received_by_arduino = np.zeros(len(y_test_reshaped))
# for i in range(X_test_reshaped.shape[0]):
#      ser = serial.Serial("COM6", 9600) 
#      this_count = send_one_datapoint_to_arduino_1( ser, i) 
#      counts_received_by_arduino[i] = this_count
#      if (this_count != 160): 
#          print('ERROR: Arduino received {:d} for datapoint {:d}'.format(this_count, i))          
#      ser.close()
#      print()
# # save also into a file
# df = pd.DataFrame({"counts_received_by_arduino" : counts_received_by_arduino})
# df.to_csv("results_counts_iniside_arduino.csv", index=False)



###############################################################################
#
# Part 2: <-------------- use with Arduino sketch "arduino_rul_prediction_v2.ino" !!!
# next function and testing of it is to be used only with 
#
###############################################################################


# (1) send one datapoint from the dataset; that is a row with 20*8 float numbers;
# here we only transferred one datapoint, we will need to loop through 
# the whole data (X_test_reshaped) later;
def send_one_datapoint_to_arduino_2( serial_port, y_predict, dp_index):
    # suppress the scientific notion!!!
    np.set_printoptions(suppress=True, precision=8)
    # data_array is a 2D numpy array
    data_array = X_test_reshaped[dp_index]
    #print("data_array: ", data_array)
    #rint("Shape of data_array: ", data_array.shape)
    #print()
    # convert the whole array to string first
    data_array = data_array.flatten()
    data_array_as_string = str(data_array)
    # suppress "\n"
    # need to use the replace function ONLY for ESP32, do not know why
    data_array_as_string = data_array_as_string.replace("\n", "")

    # encode the string by using utf-8
    data_bytes = data_array_as_string.encode('utf-8')    
    # print data_array encoded
    # print("data_array encoded to bytes: ", data_bytes)
    # print("Length of sent data_array: ", len(data_bytes)) 
    # print()  
    # send to Arduino board by writing to serial
    serial_port.write(data_bytes)
    # sleep(0.1)  // comment out!!
    # read line from serial port; a line ends with "\r\n"
    # so make sure in Arduino sketch we send everything separated by 
    # say space character " " and only at the end of a transmission include
    # "\r\n", which here will mean stop reading a line;
    #1
    #received_bytes = serial_port.readline()
    #print(received_bytes) # inference time
    #2
    received_bytes = serial_port.readline()
    print(received_bytes) # received count inside Arduino
    decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')  # here we receive the total number we send
    received_count_values_arduino = int(decoded_bytes)

    # check the Arduino received value in each array (len: 160)
    # for num in range(0, 160):
    #     received_bytes = serial_port.readline()
    #     decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
    #     print(decoded_bytes)

        # received_count_values_arduino = int(decoded_bytes)
    
    if (received_count_values_arduino != 160): 
        print('ERROR: Arduino did not receive 160 values for datapoint {:d}'.format(dp_index))
        return False
    else:  
        #3
        # print quantization scale/zero_point values
        # received_bytes = serial_port.readline()
        # print(received_bytes)
        # decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
        # print("Length of received decoded_bytes: ", len(decoded_bytes))
        # print()

        # # print y_pred quantized values
        # received_bytes = serial_port.readline()
        # decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
        # print(decoded_bytes)

        # print y_pred values
        received_bytes = serial_port.readline()
        decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
        print(decoded_bytes)
        print("Length of received decoded_bytes: ", len(decoded_bytes))
        print()
        # converting from string to float
        # split the string, so that string becomes a list of string
        str_splits = decoded_bytes.split()
        # to store float numbers
        str_to_float = []
        # iterate through the string
        received_count = 0
        for str_split in str_splits:
            str_split = float(str_split)
            y_predict[dp_index] = str_split 
            # append the float number to the list
            received_count = received_count + 1
            str_to_float.append(str_split)   
        #print("Float number list: ", str_to_float)
        #print("Float number list length: ", received_count) #len(str_to_float)
        return True




# (2) array where we will place all predictions for all datapoints
y_predict = np.zeros(len(y_test_reshaped)) # should be 3560


# (3) test the above function; before and after the call
# of the function, open Serial and then close Serial;
# uncomment the next lines to do the testing during debug:
# ser = serial.Serial("COM6", 9600) 
# send_one_datapoint_to_arduino_2( ser, y_predict, 1328) # index 
# ser.close()
# print()


# (4) go through all inputs and send them one by one to Arduino sketch that
# uses the tinyML model to make predictions;
# each datapoint sent is a list of 20*8 float numbers;
# the response is just one float number for each sent datapoint of 20*8 values;
for i in range(X_test_reshaped.shape[0]):
    print('i= {:d}'.format(i))
    ser = serial.Serial("COM5", 9600)   #"COM3" for arduino, "COM5" for ESP32, "COM6" on Cris's laptop
    arduino_received_160 = send_one_datapoint_to_arduino_2( ser, y_predict, i)
    if (arduino_received_160 != True):
        i = i - 1 # send the same datapoint one more time
    ser.close()
    print()

print(y_predict)
print()


# (5) function to calculate errors
# def cal_error(y_test_reshaped: np.ndarray, y_pred_test: np.ndarray):
#     # calculate MAE
#     mae = mean_absolute_error(y_test_reshaped, y_pred_test)
#     # calculate MSE
#     mse = mean_squared_error(y_test_reshaped, y_pred_test)
#     # calculate RMSE
#     rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_test))
#     # Calculate R2
#     r2 = r2_score(y_test_reshaped, y_pred_test)
#     return mae, mse, rmse, r2

# use above function to calculate prediction errors
rmse, mse, mae, r2, _ = ModelEvaluation.cal_error(y_test_reshaped, y_predict)
print(f"Errors-> mae: {mae:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")


# (6) create and save figure
x_ax = range(len(y_predict))
plt.scatter(x_ax, y_test_reshaped, s=1, color="blue", label="original")
plt.scatter(x_ax, y_predict, s=1, color="red", label="predicted") 
#plt.plot(x_ax, y_predict, lw=0.8, color="red", label="predicted")
#plt.scatter(x_ax, y1_test_reshaped - y_predict, s=1, color="blue", label="true - predicted")
#plt.scatter(x_ax, pow(y1_test_reshaped - y_predict, 2), s=1, color="blue", label="mse")
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.title(f'{model} ND Predictions: MAE = ' + str(round(mae, 3)))
plt.xlabel('Index in Testing Dataset')
#plt.ylabel('Cell Temp. (C)')
plt.ylabel('RUL')
plt.legend()
# plt.savefig(f'./device_result/fig_{model}_{target}_revised2.png', dpi=300)
plt.show()
plt.clf()

# save also into a file
# df = pd.DataFrame({f"y_test": y_test_reshaped, "y_predicted": y_predict})
# df.to_csv(f"./device_result/results_{model}_{target}_revised2.csv", index=False)
