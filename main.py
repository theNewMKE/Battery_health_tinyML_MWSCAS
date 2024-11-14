from read_from_file import ReadFromFile
from utils import df_info, drop_cycle_index, get_hyper_params
from model_analysis import model_analysis

# define directory path
# NMC
dir_path = "../datasets/bat_arc/SNL_NMC/SNL_NMC"
# NCA
# dir_path = "../datasets/bat_arc/SNL_NCA/SNL_NCA"

# define file path
# NMC
timeseries_file_name = ["SNL_18650_NMC_15C_0-100_0.5-1C_a_timeseries.csv"]  # a list that contains all target files
# NCA
# timeseries_file_name = ["SNL_18650_NCA_35C_0-100_0.5-2C_a_timeseries.csv"]
# NMC
cycle_file_name = ['SNL_18650_NMC_15C_0-100_0.5-1C_a_cycle_data.csv']
# NCA
# cycle_file_name = ["SNL_18650_NCA_35C_0-100_0.5-2C_a_cycle_data.csv"]

# read timeseries_file
timeseries_file = ReadFromFile(dir_path, timeseries_file_name, True)
df = timeseries_file.read_file()

# read cycle_file
cycle_file = ReadFromFile(dir_path, cycle_file_name, False)
df_cd = cycle_file.read_file()

# drop abnormal datapoints from df
drop_cycle_index(df_cd, df)
# check df info
df_info(df)

# chose target
target = "tar2"    # "tar1", "tar2", "multi_tars"

# analysis for different losses
# diff_losses = ["Huber", "LogCosh", "MeanAbsoluteError", "MeanSquaredError", "MeanAbsolutePercentageError"]

# chose model name
# change regular conv or dilated conv for different tars!!!
model_name = "TCN"  # "oneD_cnn", "cnn_gru", "text_conv_net", "TCN", "MLP"

# what to save
to_save = "smooth"  # "loss", "smooth", "sample"

# smooth or sample numbers
# numbers = [1, 3, 6, 12, 18]
numbers = [18]   # check the paper!!!

# loop times for each case
loop_times = 5  # 1, 5

# get the best hyper params for model
input_hyper_params = get_hyper_params(model_name, target)

# get analysis and output smooth/sample files
model_analysis(df, target, model_name, to_save, numbers, loop_times, input_hyper_params, save_model=True)