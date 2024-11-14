import os
import pandas as pd

# check current path
# print(os.path.abspath(os.getcwd()))


class ReadFromFile:
    """
    load file is a class, in which it has methods ->
    read_file(), load_file(), smooth_inputs, down_sample
    """
    def __init__(self, _dir_path: str, _file_name: list, _is_timeseries: bool) -> None:
        self.dir_path = _dir_path
        self.file_name = _file_name
        self.is_timeseries = _is_timeseries

    def read_file(self) -> pd.DataFrame:
        """
        get the target file names as a file list,
        load csv files from the file list as dataframes,
        and concat dataframes
        """
        target_files = []
        for dirname, _, filenames in os.walk(self.dir_path):
            for filename in filenames:
                for fn in self.file_name:
                    if filename == fn:
                        # print(os.path.join(dirname, filename))
                        target_files.append(os.path.join(dirname, filename))

        li = []
        for file in target_files:
            if os.path.exists(file):
                # read scv file into dataframe
                df = pd.read_csv(file, index_col=None, header=0)
                if self.is_timeseries:
                    print(f"Timeseries dataframe {file} loaded.")
                else:
                    print(f"Cycle dataframe {file} loaded.")
                # append all dataframes
                li.append(df)
                df = pd.concat(li, axis=0, ignore_index=True)
            else:
                print("File does not exist, please check file name again.")
        return df


    # def org_test_time(df: pd.DataFrame) -> None:
    #     """
    #     test time is accumulative data, organize this column to non-accumulative data
    #     """
    #     tmp = df['Test_Time (s)'][0]
    #     df['Test_Time (s)'] = df['Test_Time (s)'] - df['Test_Time (s)'].shift(1)
    #     df['Test_Time (s)'][0] = tmp