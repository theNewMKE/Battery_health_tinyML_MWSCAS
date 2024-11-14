import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from utils import set_font


class ModelEvaluation:
    seq_len = 0

    def __init__(self, model, X_test_reshaped, target,
                 y_test_reshaped=None, y_test=None,
                 y1_test_reshaped=None, y1_test=None,
                 y2_test_reshaped=None, y2_test=None):

        self._model = model
        if target == "tar1" or target == "tar2":
            # get predictions
            y_pred_test = self._model.predict(X_test_reshaped, verbose=0)
            # reshape prediction result to 1d array
            y_pred_test = y_pred_test.reshape((len(y_pred_test),))
            # calculate errors
            rmse_test, mse_test, mae_test, r2_test, MAPE_test = \
                ModelEvaluation.cal_error(y_test_reshaped.astype(np.float32), y_pred_test)
            self.rmse_test = rmse_test
            self.mse_test = mse_test
            self.mae_test = mae_test
            self.r2_test = r2_test
            self.MAPE_test = MAPE_test

        elif target == "multi_tars":
            # get predictions
            y1_pred_test = self._model.predict(X_test_reshaped, verbose=0)[0]
            y2_pred_test = self._model.predict(X_test_reshaped, verbose=0)[1]
            # calculate errors
            rmse_test1, mse_test1, mae_test1, r2_test1, MAPE_test1 = ModelEvaluation.cal_error(y1_test_reshaped,
                                                                                               y1_pred_test)
            rmse_test2, mse_test2, mae_test2, r2_test2, MAPE_test2 = ModelEvaluation.cal_error(y2_test_reshaped,
                                                                                               y2_pred_test)
            self.rmse_test1 = rmse_test1
            self.mse_test1 = mse_test1
            self.mae_test1 = mae_test1
            self.r2_test1 = r2_test1
            self.MAPE_test1 = MAPE_test1

            self.rmse_test2 = rmse_test2
            self.mse_test2 = mse_test2
            self.mae_test2 = mae_test2
            self.r2_test2 = r2_test2
            self.MAPE_test2 = MAPE_test2


        # plot different type of errors
        # ModelEvaluation.sorted_actual_and_pred(y_pred_test, y_test.pkl, ModelEvaluation.seq_len, is_testing=True,)
        # ModelEvaluation.sorted_actual_and_pred(y_pred_train, y_train, ModelEvaluation.seq_len, is_testing=False)
        #
        # ModelEvaluation.mae_time_step(y_pred_test, y_test.pkl, ModelEvaluation.seq_len, is_testing=True)
        # ModelEvaluation.mae_time_step(y_pred_train, y_train, ModelEvaluation.seq_len, is_testing=False)
        #
        # ModelEvaluation.mape_time_step(y_pred_test, y_test.pkl, ModelEvaluation.seq_len, is_testing=True)
        # ModelEvaluation.mape_time_step(y_pred_train, y_train, ModelEvaluation.seq_len, is_testing=False)
        #
        # ModelEvaluation.plot_true_and_pred(y_test_reshaped, y_pred_test)
        # ModelEvaluation.plot_true_and_pred(y_train_reshaped, y_pred_train, "training")

    @classmethod
    def set_seq_len(cls, seq_len: int):
        cls.seq_len = seq_len

    @staticmethod
    def cal_error(y_test_reshaped: np.ndarray, y_pred_test: np.ndarray):

        # Calculate rmse
        rmse = round(np.sqrt(mean_squared_error(y_test_reshaped, y_pred_test)), 4)

        # Calculate MSE
        mse = round(mean_squared_error(y_test_reshaped, y_pred_test), 4)

        # Calculate MAE
        mae = round(mean_absolute_error(y_test_reshaped, y_pred_test), 4)

        # Calculate R2
        r2 = round(r2_score(y_test_reshaped, y_pred_test), 4)

        # calculate MAPE
        # mean_absolute_percentage_error
        MAPE = round(mean_absolute_percentage_error(y_test_reshaped, y_pred_test), 4)

        return rmse, mse, mae, r2, MAPE


    @staticmethod
    def create_series(y_pred: np.ndarray, y: pd.Series, seq_len: int) -> pd.Series:
        """
        change the predicted type from np.ndarray to pd.Series and return the newly created pd.Series
        """
        y_pred_series = pd.Series(data=y_pred.reshape(len(y_pred), ), index=y[seq_len - 1:].index)
        return y_pred_series

    @staticmethod
    def sorted_actual_and_pred(y_pred: np.ndarray, y: pd.Series, seq_len: int, is_testing: bool,) -> None:
        """
        plot sorted actual vs. predicted values
        """
        y_pred_series = ModelEvaluation.create_series(y_pred, y, seq_len)
        plt.figure(figsize=(20, 10))
        set_font(25, "Times New Roman", 500)
        plt.plot(y[seq_len - 1:].sort_index(), '.')
        plt.plot(y_pred_series.sort_index(), '*')
        plt.xlabel('Time Steps')
        plt.ylabel('RUL')
        # plt.ylabel('Cell_Temperature (C')
        plt.legend(labels=('Actual', 'Predicted'), loc='upper left')
        if is_testing:
            plt.title('Actual testing data vs. predicted testing data')
        else:
            plt.title('Actual training data vs. predicted training data')
        plt.xlabel('Time Steps (10 seconds)')
        plt.show()

    @staticmethod
    def mae_time_step(y_pred: np.ndarray, y: pd.Series, seq_len: int, is_testing: bool) -> None:
        """
        plot mae in each time step
        """
        y_pred_series = ModelEvaluation.create_series(y_pred, y, seq_len)
        mae = np.abs(y[seq_len - 1:].sort_index() - y_pred_series.sort_index())
        plt.figure(figsize=(20, 10))
        set_font(25, "Times New Roman", 500)
        plt.plot(mae, '*')
        if is_testing:
            plt.title('MAE in each time step for testing data')
        else:
            plt.title('MAE in each time step for training data')
        plt.xlabel('Time Steps (10 seconds)')
        plt.ylabel('MAE')
        plt.show()

    @staticmethod
    def mape_time_step(y_pred: np.ndarray, y: pd.Series, seq_len: int, is_testing: bool) -> None:
        """
        plot mape in each time step
        """
        y_pred_series = ModelEvaluation.create_series(y_pred, y, seq_len)
        mape = np.abs((y[seq_len - 1:].sort_index() - y_pred_series.sort_index()) / y[seq_len - 1:].sort_index()) * 100
        # get the value from 5000 to end
        #  mape = np.abs((y[seq_len-1:].sort_index()[5000:] - y_pred_series.sort_index()[5000:]) / y[seq_len-1:].sort_index()[5000:]) * 100
        plt.figure(figsize=(20, 10))
        set_font(25, "Times New Roman", 500)
        plt.plot(mape, '*')
        if is_testing:
            plt.title('MAPE in each time step for testing data')
        else:
            plt.title('MAPE in each time step for training data')
        plt.xlabel('Time Steps (10 seconds)')
        plt.ylabel('MAPE (%)')
        plt.show()

    @staticmethod
    def plot_true_and_pred(y_true: np.ndarray, predictions: np.ndarray, s='Testing') -> None:
        """
        actually vs predicted values and the perfect 45 degree line
        """
        plt.figure(figsize=(10, 8))
        set_font(16, "Times New Roman", 500)
        plt.xlabel("Actual data")
        plt.ylabel("Predicted data")
        plt.title(s)
        # plt.scatter(y_true,predictions)

        # perfect prediction line
        plt.plot(y_true, y_true, 'r')
        # plt.plot(predictions,predictions,'*')

        # regression plot
        sns.regplot(x=y_true, y=predictions, color='blue', marker='*')
        plt.show()