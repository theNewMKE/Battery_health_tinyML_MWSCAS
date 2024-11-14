import tensorflow as tf
import os
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import load_model
# from tensorflow_model_optimization.sparsity import keras as sparsity
from utils import get_pickled_X_test_reshaped, get_pickled_y_test_reshaped
from model_evaluation import ModelEvaluation


# from Cris tinyML course week8, train_TFL_Micro_hello_world_model
def predict_tflite(tflite_model, x_test):
    # Prepare the test data
    x_test_ = x_test
    # x_test_ = x_test_.reshape((x_test.size, 1))
    x_test_ = x_test_.astype(np.float32)

    # Initialize the TFLite interpreter
    # interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # If required, quantize the input layer (from float to integer)
    input_scale, input_zero_point = input_details["quantization"]
    if (input_scale, input_zero_point) != (0.0, 0):
        x_test_ = x_test_ / input_scale + input_zero_point
        x_test_ = x_test_.astype(input_details["dtype"])

    # Invoke the interpreter
    y_pred = np.empty(len(x_test_), dtype=output_details["dtype"])

    # start_time = time.time()
    for i in range(len(x_test_)):
        interpreter.set_tensor(input_details["index"], [x_test_[i]])
        interpreter.invoke()
        y_pred[i] = interpreter.get_tensor(output_details["index"])[0]
    # end_time = time.time()
    # print(end_time-start_time)

    # If required, de-quantized the output layer (from integer to float)
    output_scale, output_zero_point = output_details["quantization"]
    if (output_scale, output_zero_point) != (0.0, 0):
        y_pred = y_pred.astype(np.float32)
        y_pred = (y_pred - output_zero_point) * output_scale

    return y_pred


def get_pred_tflite(tflite_model, x_test):
    y_pred = predict_tflite(tflite_model, x_test)
    return y_pred


def reload_model(target: str, model: str):

    # get corresponding X and y data
    X_test_reshaped = get_pickled_X_test_reshaped(target, model)
    y_test_reshaped = get_pickled_y_test_reshaped(target, model)

    # define model folder path
    model_path = 'models_test/'
    # define tf model path
    model_tf_path = f'model_{target}/'

    # load the tf model
    # tf load method
    # model_tar = tf.saved_model.load(model_path+model_tar2_tf)
    # tf keras load_model method
    model_tf = load_model(model_path+model_tf_path)

    # define no_quant_tflite and tflite paths
    # check the tf model by using the saved folder
    model_tflite_no_quan_path = f'model_{target}_no_quant.tflite'
    model_tflite_path = f'model_{target}.tflite'

    # get size
    # check the tf model by using the saved folder
    model_tflite_no_quan = os.path.getsize(model_path+model_tflite_no_quan_path)
    model_tflite = os.path.getsize(model_path+model_tflite_path)
    print(f'{model}_no_quan_TFLITE size: {model_tflite_no_quan / (1024 ** 1):.2f} KB')
    print(f'{model}_TFLITE size: {model_tflite / (1024 ** 1):.2f} KB')

    # get tf model predictions
    tf_pred = model_tf.predict(X_test_reshaped)
    tf_pred = tf_pred.reshape(len(tf_pred),)
    # calculate tf model errors
    rmse, mse, mae, r2, _ = ModelEvaluation.cal_error(y_test_reshaped.astype(np.float32), tf_pred)
    print(f"Error for tf_model, mae: {mae:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")

    # get the tflite_no_quan model predictions
    tflite_no_quan_pred = get_pred_tflite(model_path+model_tflite_no_quan_path, X_test_reshaped)
    # calculate tflite_no_quan model errors
    rmse, mse, mae, r2, _ = ModelEvaluation.cal_error(y_test_reshaped.astype(np.float32), tflite_no_quan_pred)
    print(f"Error for tflite_no_quan_model, mae: {mae:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")

    # get the tflite model predictions
    tflite_pred = get_pred_tflite(model_path+model_tflite_path, X_test_reshaped)
    # calculate tflite model errors
    rmse, mse, mae, r2, _ = ModelEvaluation.cal_error(y_test_reshaped.astype(np.float32), tflite_pred)
    print(f"Error for tflite_model, mae: {mae:.4f}, mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")


# define target and model
target = 'tar1'   # tar1, tar2
model = 'CNN'   # TCN CNN
# use the reload_model function
reload_model(target, model)

