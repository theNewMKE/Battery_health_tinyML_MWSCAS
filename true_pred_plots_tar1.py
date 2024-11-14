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


# def sorted_actual_and_pred(y_pred_series: pd.Series, y: pd.Series, tar: str,) -> None:
#     """
#     plot sorted actual vs. predicted values
#     """
#
#     plt.figure(figsize=(20, 10))
#     set_font(25, "Times New Roman", 500)
#     plt.plot(y.sort_index(), '*')
#     plt.plot(y_pred_series.sort_index(), '.')
#     # print(y_pred_series.sort_index())
#     plt.xlabel('Time Steps')
#     # plt.title('True data vs. predicted data')
#     plt.legend(labels=('True', 'Predicted'), loc='upper left')
#     if tar == 'tar1':
#         plt.ylabel('RUL')
#     else:
#         plt.ylabel('Cell_Temperature (C')
#     plt.xlabel('Time Steps (10 seconds)')
#     plt.show()


# for TCN_and_CNN tar1
dir_path = r"./pickled_data/TCN/tar1/"
file1 = dir_path + r"y_pred_test.pkl"
file2 = dir_path + r"y_test.pkl"
file3 = r"./pickled_data/df.pkl"

# for 1D CNN tar1
dir_path = r"./pickled_data/1D_CNN/tar1/"
file4 = dir_path + r"y_pred_test.pkl"

# for CNN_GRU tar1
dir_path = r"./pickled_data/CNN_GRU/tar1/"
file5 = dir_path + r"y_pred_test.pkl"

# for MLP tar1
dir_path = r"./pickled_data/CNN_GRU/tar1/"
file6 = dir_path + r"y_pred_test.pkl"

# TCN
y_pred_test_TCN = load_file(file1)
y_test_TCN = load_file(file2)
df = load_file(file3)


# CNN
y_pred_test_CNN = load_file(file4)

# CNN_GRU
y_pred_test_CNN_GRU = load_file(file5)

# MLP
y_pred_test_MLP = load_file(file6)

# print(len(dis_cap))
# print(len(y_pred_test_TCN))
seq_len = 20

# exclude the first 19 data points from the true test value
exclude_index = y_test_TCN[0:19].index
excluded_index = y_test_TCN[~y_test_TCN.index.isin(exclude_index)].index
y_test_TCN = y_test_TCN[~y_test_TCN.index.isin(exclude_index)]
y_test_overall = y_test_TCN

# TCN_
y_pred_series_TCN = create_series(y_pred_test_TCN, excluded_index, seq_len)
y_pred_series_TCN_overall = y_pred_series_TCN

# CNN
y_pred_series_CNN = create_series(y_pred_test_CNN, excluded_index, seq_len)
y_pred_series_CNN_overall = y_pred_series_CNN

# CNN_GRU
y_pred_series_CNN_GRU = create_series(y_pred_test_CNN_GRU, excluded_index, seq_len)
y_pred_series_CNN_GRU_overall = y_pred_series_CNN_GRU

# MLP
y_pred_series_MLP = create_series(y_pred_test_MLP, excluded_index, seq_len)
y_pred_series_MLP_overall = y_pred_series_MLP

# get the test index
test_index = y_pred_series_TCN.sort_index().index.tolist()


# print(y_test_TCN.sort_index())

df_cycles = df['Cycle_Index']

# get the index for the end of each cycle
last_index = {}
for i in range(len(df)):
    if df_cycles[i] not in last_index:
        last_index[df_cycles[i]-1] = i

# get the intersections of the end of cycle index with the index inside the true test data
end_cycle = set(test_index).intersection(set(last_index.values()))
end_cycle = list(end_cycle)

# get true test data from the intersect_index
y_test_TCN = y_test_TCN.sort_index()
y_test_TCN = y_test_TCN.loc[end_cycle]
print(y_test_TCN)

# get true predicted data from the intersect_index for TCN
y_pred_series_TCN = y_pred_series_TCN.sort_index()
y_pred_series_TCN = y_pred_series_TCN.loc[end_cycle]
# print(y_pred_series_TCN)

# get true predicted data from the intersect_index for CNN
y_pred_series_CNN = y_pred_series_CNN.sort_index()
y_pred_series_CNN = y_pred_series_CNN.loc[end_cycle]
# print(y_pred_series_CNN)

# get true predicted data from the intersect_index for CNN_GRU
y_pred_series_CNN_GRU = y_pred_series_CNN_GRU.sort_index()
y_pred_series_CNN_GRU = y_pred_series_CNN_GRU.loc[end_cycle]
# print(y_pred_series_CNN_GRU)

# get true predicted data from the intersect_index for MLP
y_pred_series_MLP = y_pred_series_MLP.sort_index()
y_pred_series_MLP = y_pred_series_MLP.loc[end_cycle]
# print(y_pred_series_CNN)


# plt.plot(y_test_TCN, '*')
# plt.plot(y_pred_series_TCN, '.')
# plt.show()
y_test_TCN = y_test_TCN.sort_index()
y_pred_series_TCN = y_pred_series_TCN.sort_index()
y_pred_series_CNN = y_pred_series_CNN.sort_index()
y_pred_series_CNN_GRU = y_pred_series_CNN_GRU.sort_index()
y_pred_series_MLP = y_pred_series_MLP.sort_index()

# get the capacity values
dis_cap = df['Discharge_Capacity (Ah)'].loc[end_cycle]
dis_cap = dis_cap.sort_index()
normalized_dis_cap = [(value - min(dis_cap)) / (max(dis_cap) - min(dis_cap)) for value in dis_cap]


# plot the intersect_index data for true and predicted values

plt.figure(figsize=(15, 9))
set_font(27, "Times New Roman", 500)
plt.xlabel('Battery cycles (RUL)')
plt.ylabel('Normalized discharge capacity')
plt.scatter(y_test_TCN.values, normalized_dis_cap, linewidths=12, edgecolors='blue', label='True')
plt.scatter(y_pred_series_CNN.values, normalized_dis_cap, linewidths=11, edgecolors='cyan', label='1D CNN')
plt.scatter(y_pred_series_CNN_GRU.values, normalized_dis_cap, linewidths=11, edgecolors='yellow', label='1D CNN+GRU')
plt.scatter(y_pred_series_TCN.values, normalized_dis_cap, linewidths=11, edgecolors='red', label='TCN_and_CNN')
plt.scatter(y_pred_series_MLP.values, normalized_dis_cap, linewidths=11, edgecolors='black', label='MLP')
plt.legend(loc='upper right')
plt.savefig('cap_vs_RUL.pdf')
plt.show()

# sorted_actual_and_pred(y_pred_series_TCN_overall, y_test_overall, tar='tar1')
# sorted_actual_and_pred(y_pred_series_CNN_overall, y_test_overall, tar='tar1')

plt.figure(figsize=(15, 9))
set_font(27, "Times New Roman", 500)
plt.plot(y_test_overall.sort_index(), 'b+', markersize=15, label='True')
plt.plot(y_pred_series_CNN_overall.sort_index(), 'c.', markersize=8, alpha=0.3, label='1D CNN')
plt.plot(y_pred_series_CNN_GRU_overall.sort_index(), 'y*', markersize=8, alpha=0.3, label='1D CNN+GRU')
plt.plot(y_pred_series_TCN_overall.sort_index(), 'ro', markersize=8, alpha=0.3, label='TCN_and_CNN')
plt.plot(y_pred_series_MLP_overall.sort_index(), 'k^', markersize=8, alpha=0.3, label='MLP')
# print(y_pred_series.sort_index())
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Battery cycles (RUL)')
plt.legend(loc='upper left')
plt.savefig('RUL_vs_TS.pdf')
plt.show()


plt.figure(figsize=(15, 9))
set_font(27, "Times New Roman", 500)
# plt.plot(y_test_overall.sort_index(), 'b+', markersize=15, label='True')
plt.plot(y_test_overall.sort_index() - y_pred_series_CNN_overall.sort_index(), 'c.', markersize=8, alpha=0.8, label='1D CNN')
plt.plot(y_test_overall.sort_index() - y_pred_series_CNN_GRU_overall.sort_index(), 'y*', markersize=8, alpha=0.8, label='1D CNN+GRU')
plt.plot(y_test_overall.sort_index() - y_pred_series_TCN_overall.sort_index(), 'ro', markersize=8, alpha=0.8, label='TCN_and_CNN')
plt.plot(y_test_overall.sort_index() - y_pred_series_MLP_overall.sort_index(), 'k^', markersize=8, alpha=0.8, label='MLP')
# print(y_pred_series.sort_index())
plt.xlabel('Time step (10 seconds)')
plt.ylabel('Battery cycles (RUL) difference')
plt.legend(loc='upper left')
plt.savefig('RUL_vs_TS_diff.pdf')
plt.show()