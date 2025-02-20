import torch
import numpy as np
import ctypes

# Load the compiled C++ deep learning library
lib = ctypes.CDLL("../Apex-Engine/build/libdeep_learning.so")

# Define input/output types
lib.predict.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
lib.predict.restype = ctypes.POINTER(ctypes.c_float)

def predict_lstm(input_data):
    input_array = np.array(input_data, dtype=np.float32)
    input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Call the C++ LSTM model
    result_ptr = lib.predict(input_ptr, len(input_data))

    # Convert the result back to NumPy
    result = np.ctypeslib.as_array(result_ptr, shape=(3,))
    return result
