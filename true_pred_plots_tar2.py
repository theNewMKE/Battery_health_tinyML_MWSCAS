import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from utils import set_font


def load_file(file_path: str):
    with open(file_path, 'rb') as f:
        var_name = pickle.load(f)
        return var_name


def create_series(y_pred: np.ndarray, exed_index, seq_len: int) -> pd.Series:
    """
    change the predicted type from np.ndarray to pd.Series and return the newly created pd.Series
    """
    y_pred_series = pd.Series(data=y_pred.reshape(len(y_pred), ), index=exed_index)
    return y_pred_series


def sorted_actual_and_pred(y_pred_series: pd.Series, y: pd.Series, tar: str,) -> None:
    """
    plot sorted actual vs. predicted values
    """

    plt.figure(figsize=(20, 10))
    set_font(25, "Times New Roman", 500)
    plt.plot(y.sort_index(), '*')
    plt.plot(y_pred_series.sort_index(), '.')
    print(y_pred_series.sort_index())
    plt.xlabel('Time Steps')
    plt.title('True data vs. predicted data')
    plt.legend(labels=('True', 'Predicted'), loc='upper left')
    if tar == 'tar1':
        plt.ylabel('RUL')
    else:
        plt.ylabel('Cell_Temperature (C)')
    plt.xlabel('Time Steps (10 seconds)')
    plt.show()

# for TCN_and_CNN tar2
dir_path = r"./pickled_data/TCN/tar2/"
file1 = dir_path + r"y_pred_test.pkl"
file2 = dir_path + r"y_test.pkl"
# file2 = "./pickled_data/y2_test_reshaped.pkl"


# for 1D CNN tar2
dir_path = r"./pickled_data/1D_CNN/tar2/"
file3 = dir_path + r"y_pred_test.pkl"

# for CNN+GRU tar2
dir_path = r"./pickled_data/CNN_GRU/tar2/"
file4 = dir_path + r"y_pred_test.pkl"


y_pred_test_TCN = load_file(file1)
y_test_TCN = load_file(file2)
y_pred_test_CNN = load_file(file3)
y_pred_test_CNN_GRU = load_file(file4)


seq_len = 20

# exclude the first 19 data points from the true test value
exclude_index = y_test_TCN[0:19].index
excluded_index = y_test_TCN[~y_test_TCN.index.isin(exclude_index)].index
y_test_TCN = y_test_TCN[~y_test_TCN.index.isin(exclude_index)]
y_test_overall = y_test_TCN

y_pred_series_TCN = create_series(y_pred_test_TCN, excluded_index, seq_len)
y_pred_series_TCN_overall = y_pred_series_TCN

y_pred_series_CNN = create_series(y_pred_test_CNN, excluded_index, seq_len)
y_pred_series_CNN_overall = y_pred_series_CNN

y_pred_series_CNN_GRU = create_series(y_pred_test_CNN_GRU, excluded_index, seq_len)
y_pred_series_CNN_GRU_overall = y_pred_series_CNN_GRU

plt.figure(figsize=(15, 9))
set_font(25, "Times New Roman", 500)
plt.plot(y_test_overall.sort_index(), 'b+', markersize=8, label='True')
plt.plot(y_pred_series_CNN_overall.sort_index(), 'c.', markersize=4, alpha=0.5, label='1D CNN')
plt.plot(y_pred_series_CNN_GRU_overall.sort_index(), 'y*', markersize=4, alpha=0.5, label='1D CNN+GRU')
plt.plot(y_pred_series_TCN_overall.sort_index(), 'ro', markersize=4, alpha=0.5, label='TCN_and_CNN')
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Cell temperature (°C)')
plt.legend(loc='upper left')
plt.savefig('Temp_vs_TS.pdf')
plt.show()

plt.figure(figsize=(15, 8))
set_font(25, "Times New Roman", 500)
# plt.plot(y_test_overall.sort_index(), 'b+', markersize=15, label='True')
plt.plot(y_test_overall.sort_index() - y_pred_series_CNN_overall.sort_index(), 'c.', markersize=8, alpha=0.3, label='1D CNN')
plt.plot(y_test_overall.sort_index() - y_pred_series_CNN_GRU_overall.sort_index(), 'y*', markersize=8, alpha=0.3, label='1D CNN+GRU')
plt.plot(y_test_overall.sort_index() - y_pred_series_TCN_overall.sort_index(), 'ro', markersize=8, alpha=0.3, label='TCN_and_CNN')
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Cell temperature (°C) residual')
plt.legend(loc='upper left')
plt.savefig('Temp_vs_TS_res.pdf')
plt.show()

