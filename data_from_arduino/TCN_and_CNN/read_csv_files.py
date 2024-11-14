import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import set_font
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


y1_pred_micro_TCN = pd.read_csv("results_TCN_tar1.csv")
y2_pred_micro_TCN = pd.read_csv("results_TCN_tar2.csv")


y1_pred_micro_CNN = pd.read_csv("results_CNN_tar1.csv")
y2_pred_micro_CNN = pd.read_csv("results_CNN_tar2.csv")


y1_pred_micro_MLP = pd.read_csv("results_MLP_tar1.csv")

y1_pred_micro_TCN = np.array(y1_pred_micro_TCN['y_predicted'])
y2_pred_micro_TCN = np.array(y2_pred_micro_TCN['y_predicted'])

y1_pred_micro_CNN = np.array(y1_pred_micro_CNN['y_predicted'])
y2_pred_micro_CNN = np.array(y2_pred_micro_CNN['y_predicted'])

y1_pred_micro_MLP = np.array(y1_pred_micro_MLP['y_predicted'])


def create_series(y_pred: np.ndarray, exed_index, seq_len: int) -> pd.Series:
    """
    change the predicted type from np.ndarray to pd.Series and return the newly created pd.Series
    """
    y_pred_series = pd.Series(data=y_pred.reshape(len(y_pred), ), index=exed_index)
    return y_pred_series

with open('cor_index.pkl', 'rb') as f:
    cor_index = pickle.load(f)

with open('y1_test_overall.pkl', 'rb') as f:
    y1_test_overall = pickle.load(f)

with open('y2_test_overall.pkl', 'rb') as f:
    y2_test_overall = pickle.load(f)


y1_pred_micro_TCN_series = create_series(y1_pred_micro_TCN, cor_index, 20)
y2_pred_micro_TCN_series = create_series(y2_pred_micro_TCN, cor_index, 20)

y1_pred_micro_CNN_series = create_series(y1_pred_micro_CNN, cor_index, 20)
y2_pred_micro_CNN_series = create_series(y2_pred_micro_CNN, cor_index, 20)

y1_pred_micro_MLP_series = create_series(y1_pred_micro_MLP, cor_index, 20)


plt.figure(figsize=(15, 8))
set_font(25, "Times New Roman", 500)
plt.plot(y1_test_overall.sort_index(), 'b+', markersize=15, label='True')
plt.plot(y1_pred_micro_CNN_series.sort_index(), 'c.', markersize=8, alpha=0.5, label='1D CNN')
plt.plot(y1_pred_micro_TCN_series.sort_index(), 'ro', markersize=8, alpha=0.5, label='TCN_and_CNN')
plt.plot(y1_pred_micro_TCN_series.sort_index(), 'k^', markersize=8, alpha=0.5, label='MLP')
# print(y_pred_series.sort_index())
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Battery cycles (RUL)')
plt.legend(loc='upper left')
# plt.savefig('RUL_vs_TS.pdf')
plt.show()

plt.figure(figsize=(15, 8))
set_font(25, "Times New Roman", 500)
plt.plot(y2_test_overall.sort_index(), 'b+', markersize=15, label='True')
plt.plot(y2_pred_micro_CNN_series.sort_index(), 'c.', markersize=8, alpha=0.5, label='1D CNN')
plt.plot(y2_pred_micro_TCN_series.sort_index(), 'ro', markersize=8, alpha=0.5, label='TCN_and_CNN')
# print(y_pred_series.sort_index())
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Cell temperature (°C)')
plt.legend(loc='lower left')
# plt.savefig('Temp_vs_TS.pdf')
plt.show()

# diff = y1_test_overall.sort_index()-y1_pred_micro_TCN_series.sort_index()
# nos = []
# for d in diff:
#     if d > 10:
#         nos.append(d)
#
# print(len(nos))
#
#
# diff = y2_test_overall.sort_index()-y2_predicro_TCN_series.sort_index()
# nos = []
# for d in diff:
#     if d > 0.5:
#         nos.append(d)
#
# print(len(nos))


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


y1_test_overall_array = np.array(y1_test_overall)
y2_test_overall_array = np.array(y2_test_overall)
print("TCN:")
mae, mse, rmse, r2 = cal_error(y1_test_overall_array, y1_pred_micro_TCN)
print("Errors-> mae, mse, rmse, r2:", mae, mse, rmse, r2)

mae, mse, rmse, r2 = cal_error(y2_test_overall_array, y2_pred_micro_TCN)
print("Errors-> mae, mse, rmse, r2:", mae, mse, rmse, r2)

print("CNN:")
mae, mse, rmse, r2 = cal_error(y1_test_overall_array, y1_pred_micro_CNN)
print("Errors-> mae, mse, rmse, r2:", mae, mse, rmse, r2)

mae, mse, rmse, r2 = cal_error(y2_test_overall_array, y2_pred_micro_CNN)
print("Errors-> mae, mse, rmse, r2:", mae, mse, rmse, r2)