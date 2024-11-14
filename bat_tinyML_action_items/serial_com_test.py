# pip install serial
import serial
import numpy as np
import pickle
# if you need to calculate the errors->
# pip install scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# load data function
def load_pickled_data(file_path):
  with open(file_path, 'rb') as f:
      return pickle.load(f)

# define data path
X_path = './pickled_data/X_test_reshaped.pkl'
y1_path = './pickled_data/y1_test_reshaped.pkl'
y2_path = './pickled_data/y2_test_reshaped.pkl'

# load data
X_test_reshaped = load_pickled_data(X_path)
y1_test_reshaped = load_pickled_data(y1_path)
y2_test_reshaped = load_pickled_data(y2_path)

print(f"The total input size, input type, and input dtype: "
      f"{X_test_reshaped.shape}, {type(X_test_reshaped)}, {X_test_reshaped.dtype}")
print(f"The total output size, output type, and output dtype for tar1: "
      f"{y1_test_reshaped.shape}, {type(y1_test_reshaped)}, {y1_test_reshaped.dtype}")
print(f"The total output size, output type, and output dtype for tar2: "
      f"{y2_test_reshaped.shape}, {type(y2_test_reshaped)}, {y2_test_reshaped.dtype}")


print("The shape of the first input :", X_test_reshaped[0].shape)
print()

# if you want to check the first input
# print("The first input:", X_test_reshaped[0])


# if you want to loop through all inputs
# for i in range(X_test_reshaped.shape[0]):
#     print(X_test_reshaped[i])
#     print('--------------------')

# define send array size and received array size
ARRAY_SIZE = 20*8
BUFFER_SIZE = 20*8

# establish serial connection with Arduino
ser = serial.Serial("COM3", 9600)  # port number

# sleep for 2 seconds if necessary
# time.sleep(2)

# data_array is a 2d numpy array
data_array = X_test_reshaped[0]

print("data_array: ", data_array)
print()

# float data cannot be converted to byte by using bytes() func directly
# so convert the whole array to string first
data_array = str(data_array)
# encode the string by using utf-8
data_bytes = data_array.encode('utf-8')   # float communication sometimes doesn't work

# print data_array encoded
print("data_array encoded to bytes: ", data_bytes)
print()

# write to serial
ser.write(data_bytes)


# I forget how I come up this: 2*(ARRAY_SIZE//BUFFER_SIZE)
# but it seems working for most of the time
# this will give back the 160 numbers as a string
# but sometimes I only get 156 numbers back as a string
for i in range(0, 2*(ARRAY_SIZE//BUFFER_SIZE)):
    received_bytes = ser.readline()
    decoded_bytes = received_bytes.decode("utf-8").strip('\r\n')
    print(decoded_bytes)
print()
# here we only transferred one data, we will need to loop through the whole data (X_test_reshaped)

# converting from string to float
# split the string
str_splits = decoded_bytes.split()
# to store float numbers
str_to_float = []
# iterate through the string
for str_split in str_splits:
    str_split = float(str_split)
    # append the float number to the list
    str_to_float.append(str_split)

print("Float number list: ", str_to_float)
print("Float number list length: ", len(str_to_float))

# close serial connection
ser.close()
print()

# function for calculate the errors
def cal_error(y_test_reshaped: np.ndarray, y_pred_test: np.ndarray):

    # calculate MAE
    mae = mean_absolute_error(y_test_reshaped, y_pred_test)

    # calculate MSE
    mse = mean_squared_error(y_test_reshaped, y_pred_test)

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_test))

    # Calculate R2
    r2 = r2_score(y_test_reshaped, y_pred_test)

    return mae, mse, rmse, r2

# an example of getting prediction errors
# if you could get the predictions from the board
# you could use the cal_error function to calculate the errors->
# y_predict = np.zeros(len(y1_test_reshaped))
# mae, mse, rmse, r2 = cal_error(y1_test_reshaped, y_predict)
# print("Errors-> mae, mse, rmse, r2:", mae, mse, rmse, r2)

