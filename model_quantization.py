import os
import tensorflow as tf
import numpy as np
from utils import get_pickled_X_test_reshaped

def full_int_quant(X_test_reshaped, converter, MODEL_TFLITE) -> None:
    # quantize weights and input from float32 to int8
    # representative_dataset
    def representative_dataset():
        for i in range(len(X_test_reshaped)):
            # from the example
            # yield ([x_train[i].reshape(1, 1)])
            yield ([X_test_reshaped[i].reshape(1, 20, 8).astype(np.float32)])


    # full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset
    model_tflite = converter.convert()
    # Save the tflite model to disk
    open(MODEL_TFLITE, "wb").write(model_tflite)


def float16_quant(converter, MODEL_TFLITE) -> None:
    # float16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    model_tflite = converter.convert()
    open(MODEL_TFLITE, "wb").write(model_tflite)


def dynamic_range_quant(converter, MODEL_TFLITE) -> None:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_tflite = converter.convert()
    open(MODEL_TFLITE, "wb").write(model_tflite)


def model_quantization(target: str, model: str):
    # save MODEL_TF, MODEL_TFLITE_NO_QUANT, MODEL_TFLITE,
    MODELS_DIR = "models_test/"
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    # from tf keras save
    MODEL_TF = MODELS_DIR + 'model_' + target
    # from .h5 model  save
    # MODEL_TF = MODELS_DIR + "model_" + target + ".h5"
    # from keras model save
    # MODEL_TF = MODELS_DIR + "model_" + target + ".keras"

    # define no quant model path
    MODEL_TFLITE_NO_QUANT = MODELS_DIR + f'model_{target}_no_quant.tflite'

    # load the TF keras model and convert it to TFLITE model
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)

    # load the TF.h5 model and convert it to TFLITE model
    # MODEL_TF = load_model(MODEL_TF)
    # converter = tf.lite.TFLiteConverter.from_keras_model(MODEL_TF)

    # save the no_quant_tflite model
    model_no_quant_tflite = converter.convert()
    open(MODEL_TFLITE_NO_QUANT, "wb").write(model_no_quant_tflite)

    # define the tflite model path
    MODEL_TFLITE = MODELS_DIR + f'model_{target}.tflite'

    # load corresponding X_test_reshaped
    X_test_reshaped = get_pickled_X_test_reshaped(target, model)

    # use full_int_quant
    full_int_quant(X_test_reshaped, converter, MODEL_TFLITE)

    # float16_quant
    # float16_quant(converter, MODEL_TFLITE)

    # dynamic_range_quant
    # dynamic_range_quant(converter, MODEL_TFLITE)


# define target and model
target = 'tar2'   # tar1
model = 'CNN'   # TCN
# use the model_quantization function
model_quantization(target, model)
