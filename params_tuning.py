import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from read_from_file import ReadFromFile
from utils import data_prep, drop_cycle_index, data_split, create_time_steps, record_errors
from build_train_models import OneDCnn, TextConvNet, CnnGru, TempConvNet



# define director path
dir_path = '../datasets/bat_arc/SNL_NMC/SNL_NMC'
# NCA a = '../datasets/bat_arc/SNL_NCA/SNL_NCA'
# dir_path = '../datasets/bat_arc/SNL_LFP/SNL_LFP'


# define file path
timeseries_file_name = ['SNL_18650_NMC_15C_0-100_0.5-1C_a_timeseries.csv']
# timeseries_file_name = ['SNL_18650_NCA_15C_0-100_0.5-1C_a_timeseries.csv']
# timeseries_file_name = ['SNL_18650_LFP_15C_0-100_0.5-1C_a_timeseries.csv']
# file_name = ['SNL_18650_NMC_15C_0-100_0.5-1C_b_timeseries.csv']

cycle_file_name = ['SNL_18650_NMC_15C_0-100_0.5-1C_a_cycle_data.csv']
# cycle_file_name = ['SNL_18650_NCA_15C_0-100_0.5-1C_a_cycle_data.csv']
# cycle_file_name = ['SNL_18650_LFP_15C_0-100_0.5-1C_a_cycle_data.csv']

# read timeseries_file
timeseries_file = ReadFromFile(dir_path, timeseries_file_name, True)
df = timeseries_file.read_file()

# read cycle_file
cycle_file = ReadFromFile(dir_path, cycle_file_name, False)
df_cd = cycle_file.read_file()

# drop abnormal datapoints from df
drop_cycle_index(df_cd, df)
# check df info
# df_info(df)

# make a copy of df
df_copy = df

# use down sample
sample_number = 3
df = data_prep(df, False, sample_number=sample_number)
print(f"You are using down sampling, the down sampling number is {sample_number}")


# train test split
target = "multi_tars"    # or tar2 or multi_tars

# for one tar
# X_train, X_val, X_test, y_train, y_val, y_test.pkl, scaler = data_split(df=df, with_cap_inputs=True,
#                                                                     tar=target, random_state=101,)

# for multi_tars
X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, scaler = \
    data_split(df=df, with_cap_inputs=True, tar=target, random_state=101,)

# create time step for data
seq_len = 20
# X_train_reshaped, X_val_reshaped, X_test_reshaped, y_train_reshaped, y_val_reshaped, y_test_reshaped = \
#     create_time_steps(seq_len, X_train, X_val, X_test, y_train, y_val, y_test.pkl)

# for multi_tars
X_train_reshaped, X_val_reshaped, X_test_reshaped, y1_train_reshaped, y1_val_reshaped, y1_test_reshaped = \
    create_time_steps(seq_len, X_train, X_val, X_test, y1_train, y1_val, y1_test)

_, _, _, y2_train_reshaped, y2_val_reshaped, y2_test_reshaped = \
    create_time_steps(seq_len, X_train, X_val, X_test, y2_train, y2_val, y2_test)

num_samples_tar1 = len(y1_train_reshaped)
num_samples_tar2 = len(y2_train_reshaped)

# model_name = "text_conv_net"
# model_name = "cnn_gru"
model_name = "TCN_and_CNN"
hyper_params = {"filters1": [64],
                "kernel_size1": [2, 3, 4, 5],
                # "units": [8, 16, 32, 64],
                # "filters2": [8, 16, 32, 64],
                # "kernel_size2": [2, 3, 4, 5],
                "alpha": 0.01,
                "loss": "Huber",
                "delta": [0.1, 0.01],
                "lr": [0.001],
                "weight1": [0.35, 0.3, 0.25, 0.2]
                }
a = []
b = []
c = []
d = []
e = []
i = 1

file_name = f"params_tuning/params_tuning_{model_name}_{target}_NMC.txt"

for filters1 in hyper_params["filters1"]:
    for kernel_size1 in hyper_params["kernel_size1"]:
        # for kernel_size2 in hyper_params["kernel_size2"]:
        for delta in hyper_params["delta"]:
            # for unit in hyper_params["units"]:
            for lr in hyper_params["lr"]:
                for weight in hyper_params["weight1"]:
                    input_hyper_params = {"filters1": filters1,
                                          "kernel_size1": kernel_size1,
                                          # "units": unit,
                                          # "kernel_size2": kernel_size2,
                                          "alpha": 0.01,
                                          "loss": "Huber",
                                          "delta": 0.1,
                                          "lr": lr}
                    weight_matrix_tar1 = np.full((num_samples_tar1, 1), weight)
                    weight_matrix_tar2 = np.full((num_samples_tar2, 1), 1-weight)

                    print(f"The {i} experiment-> filters: {filters1}, kernel_size1: {kernel_size1}, "
                          f"delta: {delta}, lr: {lr}, weight1: {weight}")
                    # model_init = OneDCnn(model_name, X_train_reshaped, input_hyper_params)
                    # model_init = TextConvNet(model_name, X_train_reshaped, input_hyper_params)
                    # model_init = CnnGru(model_name, X_train_reshaped, input_hyper_params)
                    model_init = TempConvNet(model_name, X_train_reshaped, input_hyper_params, multi_tars=True)
                    model = model_init.output_model()

                    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

                    # Train the model with the EarlyStopping callback
                    history = model.fit(X_train_reshaped,
                                        [y1_train_reshaped,
                                         y2_train_reshaped],
                                        epochs=70,
                                        batch_size=32,
                                        validation_data=(X_val_reshaped, [y1_val_reshaped, y2_val_reshaped]),
                                        callbacks=[early_stopping],
                                        verbose=0,
                                        sample_weight=[weight_matrix_tar1, weight_matrix_tar2])

                    y_pred_test = model.predict(X_test_reshaped, verbose=0)

                    # rmse
                    print()
                    # rmse_test = np.sqrt(mean_squared_error(y_test_reshaped, y_pred_test))
                    rmse_test1 = np.sqrt(mean_squared_error(y1_test_reshaped, y_pred_test[0]))
                    rmse_test2 = np.sqrt(mean_squared_error(y2_test_reshaped, y_pred_test[1]))
                    # Calculate MSE
                    # mse_test = mean_squared_error(y_test_reshaped, y_pred_test)
                    mse_test1 = mean_squared_error(y1_test_reshaped, y_pred_test[0])
                    mse_test2 = mean_squared_error(y2_test_reshaped, y_pred_test[1])
                    # Calculate MAE
                    # mae_test = mean_absolute_error(y_test_reshaped, y_pred_test)
                    mae_test1 = mean_absolute_error(y1_test_reshaped, y_pred_test[0])
                    mae_test2 = mean_absolute_error(y2_test_reshaped, y_pred_test[1])
                    # Calculate R2
                    # r2_test = r2_score(y_test_reshaped, y_pred_test)
                    r2_test1 = r2_score(y1_test_reshaped, y_pred_test[0])
                    r2_test2 = r2_score(y2_test_reshaped, y_pred_test[1])
                    # calculate MAPE
                    # MAPE_test = mean_absolute_percentage_error(y_test_reshaped, y_pred_test)
                    MAPE_test1 = mean_absolute_percentage_error(y1_test_reshaped, y_pred_test[0])
                    MAPE_test2 = mean_absolute_percentage_error(y2_test_reshaped, y_pred_test[1])

                    # a.append(rmse_test)
                    # b.append(mse_test)
                    # c.append(mae_test)
                    # d.append(r2_test)
                    # e.append(MAPE_test)
                    a.append((rmse_test1, rmse_test2))
                    b.append((mse_test1, mse_test2))
                    c.append((mae_test1, mae_test2))
                    d.append((r2_test1, r2_test2))
                    e.append((MAPE_test1, MAPE_test2))
                    file = open(file_name, "a+")
                    file.write(f"\nThe {i} experiment\n")
                    file.write(f"filters: {filters1}, kernel_size1: {kernel_size1}, "
                               f"delta: {delta}, lr: {lr}, weight1: {weight}\n")
                    # file.write(f"rmse_test: {rmse_test}, mse_test: {mse_test}, mae_test: {mae_test}, "
                    #            f"r2_test: {r2_test}, MAPE_test: {MAPE_test}\n")
                    file.write(f"rmse_test1: {rmse_test1}, mse_test1: {mse_test1}, mae_test1: {mae_test1}, "
                               f"r2_test1: {r2_test1}, MAPE_test1: {MAPE_test1}\n")
                    file.write(f"rmse_test2: {rmse_test2}, mse_test2: {mse_test2}, mae_test: {mae_test2}, "
                               f"r2_test2: {r2_test2}, MAPE_test2: {MAPE_test2}\n")
                    # file.write(f"val_loss mean:{np.mean(history.history['val_loss'])}\n")
                    # file.write(f"val_loss std:{np.std(history.history['val_loss'])}\n\n")
                    file.flush()
                    i = i+1


print()
# rmse_test_smallest = np.argmin(a)+1
rmse_test_smallest = np.argmin(a, 0)+1
print(a, "\n", rmse_test_smallest)

mse_test_smallest = np.argmin(b, 0)+1
print(b, "\n", mse_test_smallest)

mae_test_smallest = np.argmin(c, 0)+1
print(c, "\n", mae_test_smallest)

r2_test_largest = np.argmax(d, 0)+1
print(d, "\n", r2_test_largest)

MAPE_test_smallest = np.argmin(e, 0)+1
print(e, "\n", MAPE_test_smallest)


file = open(file_name, "a+")
file.write(f"rmse_test_smallest: {rmse_test_smallest}, mse_test_smallest:{mse_test_smallest}, "
           f"mae_test_smallest: {mae_test_smallest}, r2_test_largest:{r2_test_largest}, "
           f"MAPE_test_smallest: {MAPE_test_smallest}")
file.flush()


# best params for one_d_cnn NMC tar 1: down smapling 3
# The 184 experiment
# filters1: 16, kernel_size1:4, filters2:8, kernel_size2:3, lr:0.01
# rmse_test:3.7868551544771694, mse_test:14.340271960990307, mae_test:2.974871614316456, r2_test:0.9994123659532805, MAPE_test:0.2499483670557871
# val_loss mean:0.2085464642693599
# val_loss std:0.14039715932008787

# best params for one_d_cnn NMC tar 2: down smapling 3
# The 16 experiment
# filters1: 8, kernel_size1:2, filters2:16, kernel_size2:4, lr:0.01
# rmse_test:0.5919077362881965, mse_test:0.35035476827781714, mae_test:0.3443532724591698, r2_test:0.9077181255667341, MAPE_test:0.01909989074182599
# val_loss mean:0.006324015947757289
# val_loss std:0.0028365856053010853



# best params for text_conv NMC tar 1: down smapling 3
# from the second round
# The best MAE:
# The 75 experiment
# filters1: 8, kernel_size1: 3, kernel_size2: 3, filters2: 8, kernel_size3: 2, kernel_size4: 5, lr:0.01
# rmse_test: 4.1166156752894, mse_test: 16.9465246180384, mae_test: 3.1594695598140894, r2_test: 0.9993055672259064, MAPE_test: 0.30069839291323847
# val_loss mean:0.233960944124394
# val_loss std:0.3871316313535805
#
#
# The best MAPE:
# The 166 experiment
# filters1: 16, kernel_size1: 3, kernel_size2: 2, filters2: 8, kernel_size3: 4, kernel_size4: 2, lr:0.01
# rmse_test: 6.418959227445278, mse_test: 41.20303756360488, mae_test: 4.601337829887156, r2_test: 0.9983115865747527, MAPE_test: 0.09776641491512103
# val_loss mean:0.4058886696584523
# val_loss std:0.48421935506526886

# from the 3rd round
# The 72 experiment
# filters1: 8, kernel_size1: 3, kernel_size2: 3, kernel_size3: 2filters2: 8, kernel_size4: 2, kernel_size5: 5, kernel_size6: 5, lr:0.01
# rmse_test: 4.461254250816742, mse_test: 19.90278949043045, mae_test: 3.5211537838794507, r2_test: 0.9991844257374559, MAPE_test: 0.19061630268824678
# val_loss mean:0.19654881612708172
# val_loss std:0.17602952498879113


# The best params for text_conv NMC tar 2: down smapling 3
# The 271 experiment
# filters1: 64, kernel_size1: 4, filters2: 16, kernel_size2: 2, lr: 0.01
# rmse_test: 0.6849482138555005, mse_test: 0.46915405566384055, mae_test: 0.43287420578262875, r2_test: 0.876426926148477, MAPE_test: 0.023861287350706122
# val_loss mean:0.007841740584661883
# val_loss std:0.0035998106130866836

# Second best params for text_conv NMC tar 2: down smapling 3
# The 147 experiment
# filters1: 32, kernel_size1: 2, filters2: 8, kernel_size2: 3, lr: 0.01
# rmse_test: 0.7143913564520149, mse_test: 0.5103550101733498, mae_test: 0.4039935521933249, r2_test: 0.8655747795392938, MAPE_test: 0.022750491252356016
# val_loss mean:0.008014368217890009
# val_loss std:0.004202973459257049