import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

def df_info(df: pd.DataFrame) -> None:
    print()
    print(df.info())
    # print(df.describe)
    print(df.columns.to_list())
    print('Column length:', len(df.columns.to_list()))
    print()


def data_prep(df: pd.DataFrame, is_smooth_inputs: bool, time_step_smooth=1, sample_number=1) -> pd.DataFrame:
    """
    :param df: pd.DataFrame
    :param is_smooth_inputs: bool
    :param time_step_smooth: int
    :param sample_number: int
    :return: pd.DataFrame
    """
    # smooth inputs or down sample dataset
    if is_smooth_inputs:
        smooth_inputs(df, time_step_smooth)
    else:
        df = down_sample(df, sample_number)
        return df


def smooth_inputs(df: pd.DataFrame, time_step_smo: int) -> None:
    """
    smooth inputs by getting the mean of desired time steps
    """
    # 'Current (A)', 'Voltage (V)',
    col_names = ['Current (A)', 'Voltage (V)', 'Charge_Capacity (Ah)',
                 'Discharge_Capacity (Ah)', 'Charge_Energy (Wh)', 'Discharge_Energy (Wh)',
                 'Environment_Temperature (C)', 'Cell_Temperature (C)']
    for col_name in col_names:
        df[col_name] = df.groupby(df.index // time_step_smo)[col_name].transform('mean')


def down_sample(df: pd.DataFrame, time_step_ds) -> pd.DataFrame:
    """
    :param df: pd.DataFrame
    :param time_step_ds: int
    :return: pd.DataFrame
    """
    return df.iloc[::time_step_ds]


def set_font(size: int, family: str, weight: int) -> None:
    '''
    set font size, family, and weight
    '''
    # font size and family: https://stackabuse.com/change-font-size-in-matplotlib/
    plt.rcParams['font.size'] = size
    plt.rcParams['font.family'] = family
    plt.rcParams[
        'font.weight'] = weight  # Font-weight defines the thinness or thickness of a font. The ranges are 100 to 900. Normal font is 400


def plot_subfig(figsize: tuple, item: list, labels: list, linewidth=1.5, ) -> None:
    '''
    plot subfigures
    '''
    fig, axs = plt.subplots(len(item), figsize=figsize)

    for i in range(len(item)):
        axs[i].plot(item[i], linewidth=1.5)
        axs[i].set_xlabel(labels[0])
        axs[i].set_ylabel(labels[i + 1])
        # axs[i].set_xlim(labels[-1])
        fig.tight_layout()


def plot_cap_cycle_fig(df_cd: pd.DataFrame) -> None:
    set_font(10, "Times New Roman", 400)
    plt.figure(figsize=(8, 4))
    plt.xlabel('Cycle')
    plt.ylabel('Capacity (Ah)')
    plt.title('Capacity vs. cycles')
    plt.plot(df_cd['Cycle_Index'], df_cd['Discharge_Capacity (Ah)'])
    plt.show()


def drop_cycle_index(df_cd: pd.DataFrame, df: pd.DataFrame):
    # need to check figure from df_cd for capacity values
    # for NMC
    index = df_cd[(df_cd['Discharge_Capacity (Ah)'] > 2.7) | (df_cd['Discharge_Capacity (Ah)'] < 1.6)].index
    # for LFP
    # index = df_cd[ (df_cd['Discharge_Capacity (Ah)'] > 1.1) | (df_cd['Discharge_Capacity (Ah)'] < 0.94) ].index
    # for NCA
    # index = df_cd[(df_cd['Discharge_Capacity (Ah)'] > 3.158) | (df_cd['Discharge_Capacity (Ah)'] < 1.737)].index
    # print(index)
    Cycle_Index_del = df_cd.iloc[index]['Cycle_Index']
    print("We need to also delete this cycle index in original dataframe:", Cycle_Index_del.values)
    # drop those index from df_cd
    df_cd.drop(index, inplace=True)
    # drop those index from df
    for _, item in enumerate(Cycle_Index_del.values):
        # get the index of df['Cycle_Index'] == item and then drop it
        df.drop(df[df['Cycle_Index'] == item].index, inplace=True)
    # drop also the Data_time column
    df.drop(columns=['Date_Time'], axis=1, inplace=True)


def data_split(df: pd.DataFrame, tar: str, random_state: int, test_size1=0.25, test_size2=0.3):

    # create input data without two targets
    df_new = df.drop(columns=['Cycle_Index', 'Cell_Temperature (C)'], axis=1)

    # create two targets
    tar1 = df['Cycle_Index']
    tar2 = df['Cell_Temperature (C)']

    # check on two targets
    # print("Target 1: Cycle_Index:\n", tar1.values)
    # print()
    # print("Target 2: Cell_Temperature (C):\n", tar2.values)

    # tran_test_split
    X = df_new
    # tar1: cycle index,  tar2: cell temperature, multi_tars: tar1 and tar2
    if tar == "tar1" or tar == "tar2":
        target = tar1 if tar == "tar1" else tar2
        X_train, X_tmp, y_train, y_tmp = train_test_split(X.values, target,
                                                          test_size=test_size1, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,
                                                        test_size=test_size2, random_state=random_state)
        X_train, X_val, X_test, scaler = use_scaler(X_train, X_val, X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler

    elif tar == "multi_tars":
        X_train, X_tmp, y1_train, y1_tmp, y2_train, y2_tmp = train_test_split(
            X.values, tar1, tar2, test_size=test_size1, random_state=random_state)
        X_val, X_test, y1_val, y1_test, y2_val, y2_test = train_test_split(
            X_tmp, y1_tmp, y2_tmp, test_size=test_size2, random_state=random_state)
        X_train, X_val, X_test, scaler = use_scaler(X_train, X_val, X_test)
        return X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, scaler
    else:
        raise ValueError(f"Invalid target name: {tar}!")


def use_scaler(X_train, X_val, X_test):
    scaler = MinMaxScaler()
    # fit and transform for X_train
    X_train = scaler.fit_transform(X_train)
    # only transform X_test
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def get_hyper_params(model_name, target):
    input_hyper_params = None
    if model_name == "oneD_cnn" and target == "tar1":
        # one_d_cnn for tar1
        input_hyper_params = {"filters1": 32, "kernel_size1": 4, "filters2": 64, "kernel_size2": 2,
                              "loss": "Huber", "lr": 0.001, "delta": 0.1}

    elif model_name == "oneD_cnn" and target == "tar2":
        # one_d_cnn for tar2
        input_hyper_params = {"filters1": 8, "kernel_size1": 2, "filters2": 16, "kernel_size2": 4,
                              "loss": "Huber", "lr": 0.01, "delta": 0.1}

    elif model_name == "cnn_gru" and target == "tar1":
        # cnn_gru for tar1
        input_hyper_params = {"filters1": 32, "kernel_size1": 4, "units": 64,
                              "loss": "Huber", "lr": 0.001, "delta": 0.1}

    elif model_name == "cnn_gru" and target == "tar2":
        # cnn_gru for tar2
        input_hyper_params = {"filters1": 64, "kernel_size1": 4, "units": 16,
                              "loss": "Huber", "lr": 0.01, "delta": 0.1}

    elif model_name == "text_conv_net" and target == "tar1":
        # text_conv_net for tar1
        input_hyper_params = {"filters1": 16, "kernel_size1": 3, "kernel_size2": 2,
                              "loss": "Huber", "lr": 0.001, "delta": 0.1}

    elif model_name == "text_conv_net" and target == "tar2":
        # text_conv_net for tar2
        input_hyper_params = {"filters1": 64, "kernel_size1": 2, "kernel_size2": 5,
                              "loss": "Huber", "lr": 0.01, "delta": 0.1}

    elif model_name == "TCN" and target == "tar1":
        # tcn for tar1
        input_hyper_params = {"filters1": 16, "kernel_size1": 4,  "loss": "Huber", "lr": 0.01, "delta": 0.1}

    elif model_name == "TCN" and target == "tar2":
        # tcn for tar2
        # original hyper_params
        input_hyper_params = {"filters1": 32, "kernel_size1": 2, "loss": "Huber", "lr": 0.001, "delta": 0.1}

    elif model_name == "MLP" and target == "tar1":
        # mlp for tar1
        input_hyper_params = {"filters1": None, "kernel_size1": None, "n1": 64, "n2": 32, "n3": 0,
                              "loss": "Huber", "lr": 0.001, "delta": 0.1}

    elif model_name == "MLP" and target == "tar2":
        # mlp for tar2
        input_hyper_params = {"filters1": None, "kernel_size1": None, "n1": 64, "n2": 32, "n3": 0,
                              "loss": "Huber", "lr": 0.01, "delta": 0.1}

    elif model_name == "TCN" and target == "multi_tars":
        # TCN_and_CNN for multi_tars
        input_hyper_params = {"filters1": 64, "kernel_size1": 2,
                              "loss": "Huber", "lr": 0.001, "delta": 0.01, "weight": 0.3}



    else:
        raise KeyError("Invalid model_name or target.")

    return input_hyper_params


def create_time_steps(seq_len: int, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):

    # define sequence length as 20
    seq_len = 20

    # get the shape for X_train and X_test
    X_train_shape = X_train.shape
    X_val_shape = X_val.shape
    X_test_shape = X_test.shape

    # create sequences for X_train, X_val, X_test, y_train, y_val, and y_test.pkl
    X_train_reshaped, y_train_reshaped = get_time_steps(seq_len, X_train_shape, X_train, y_train.values)
    X_val_reshaped, y_val_reshaped = get_time_steps(seq_len, X_val_shape, X_val, y_val.values)
    X_test_reshaped, y_test_reshaped = get_time_steps(seq_len, X_test_shape, X_test, y_test.values)
    print()
    print("X_train_reshaped: ", X_train_reshaped.shape)
    print("X_val_reshaped: ", X_val_reshaped.shape)
    print("X_test_reshaped: ", X_test_reshaped.shape)
    print("y_train_reshaped: ", y_train_reshaped.shape)
    print("y_val_reshaped: ", y_val_reshaped.shape)
    print("y_test_reshaped:", y_test_reshaped.shape)
    print("kernel size for 1D CNN:", X_train_reshaped.shape[1:])
    print()
    return X_train_reshaped, X_val_reshaped, X_test_reshaped, y_train_reshaped, y_val_reshaped, y_test_reshaped


def get_time_steps(seq_len: int, X_shape: tuple, X: np.ndarray, y: np.ndarray):
    '''
    create time step inputs for NN model
    '''
    # the number of sequences
    # the first 19 targets are not predicted
    num_sequences = X_shape[0] - seq_len + 1

    # create an empty array for the sequences
    # shape for X is a 3d array (num of sequences, sequence length ,length of inputs)
    X_reshaped = np.zeros((num_sequences, seq_len, X_shape[1]))
    y_reshaped = np.zeros(num_sequences)

    # create the new array with sequences
    for i in range(num_sequences):
        X_reshaped[i] = X[i:i + seq_len]
        y_reshaped[i] = y[i + seq_len - 1]

    # return X_reshaped and y_reshaped
    return X_reshaped, y_reshaped


def create_weight_matrix(sample_len, weight):
    weight_matrix = np.full((sample_len, 1), weight)
    return weight_matrix


def record_errors(error_li: list, model_eva, target: str) -> list:
    # get the error values
    if target == "tar1" or target == "tar2":
        # for one tar
        tmp = [model_eva.mae_test, model_eva.mse_test, model_eva.rmse_test, model_eva.r2_test, model_eva.MAPE_test]
        for j in range(len(error_li)):
            error_li[j].append(tmp[j])
        return error_li
    elif target == "multi_tars":
        # for multi_tars
        tmp1 = [model_eva.mae_test1, model_eva.mse_test1, model_eva.rmse_test1, model_eva.r2_test1, model_eva.MAPE_test1]
        tmp2 = [model_eva.mae_test2, model_eva.mse_test2, model_eva.rmse_test2, model_eva.r2_test2, model_eva.MAPE_test2]
        for j in range(len(error_li)):
            error_li[j].append((tmp1[j], tmp2[j]))
        return error_li
    else:
        raise KeyError("The target can only be tar1, tar2, or multi_tars.")


def save_errors_to_file(error_li: list, number: int, to_save: str, target: str, input_hyper_params: dict) -> None:
    print()
    error_mean = np.round(np.mean(error_li, axis=1), 2)
    error_std = np.round(np.std(error_li, axis=1), 2)
    print("Error means->", error_mean)
    print("Error stds->", error_std)
    # write into text file as append mode
    file = open(f"./diff_{to_save}_nos_{target}/{to_save}_no{number}.txt", "a+")
    write_into_file(file, error_mean, error_std, input_hyper_params)


def write_into_file(file, error_mean: float, error_std: float, input_hyper_params: dict) -> None:
    file.write("Test errors->\n")
    file.write("[mae1, mae2], [mse1, mse2], [rmse1, rmse2], [r2_1, r2_2], [mape1, mape2]->\n")
    file.write("Error means:\n")
    file.write(str(error_mean))
    file.write("\n")
    file.write("Error stds:\n")
    file.write(str(error_std))
    file.write("\n")
    file.write(str(input_hyper_params))

def load_pickled_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def get_pickled_X_test_reshaped(target: str, model: str):
    if target == 'tar1':
        if model == 'CNN':
            X_path = 'pickled_data/X_test_reshaped1_CNN.pkl'
        elif model == 'TCN':
            X_path = 'pickled_data/X_test_reshaped1_TCN.pkl'
        else:
            X_path = None
    elif target=='tar2':
        X_path = 'pickled_data/X_test_reshaped2.pkl'
    else:
        X_path = None
    X_test_reshaped = load_pickled_data(X_path)
    return X_test_reshaped

def get_pickled_y_test_reshaped(target: str, model: str):
    if target == 'tar1':
        y_path = 'pickled_data/y1_test_reshaped.pkl'
    elif target == 'tar2':
        y_path = 'pickled_data/y2_test_reshaped.pkl'
    y_test_reshaped = load_pickled_data(y_path)
    return y_test_reshaped


# load the model, has not been used in any code!
def load_model(model_name: str, target: str):
    loaded_model = tf.keras.models.load_model(f"model_scaler_saved/{model_name}_model_{target}")

    # Get the weights of the loaded model
    loaded_weights = loaded_model.get_weights()
    return loaded_model, loaded_weights





