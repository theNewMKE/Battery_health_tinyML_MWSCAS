import pandas as pd
from utils import df_info, data_prep, plot_cap_cycle_fig,\
    data_split, create_time_steps, record_errors, save_errors_to_file, create_weight_matrix
from build_train_models import TrainModel
from model_evaluation import ModelEvaluation
import os


def model_analysis(df: pd.DataFrame, target: str, model_name: str, to_save: str,
                   numbers: list, loop_times: int, input_hyper_params: dict, save_errors=True, save_model=False):

    def save_model_and_scaler(trained_model, model_name: str, scaler, target: str):
        # save MODEL_TF, MODEL_NO_QUANT_TFLITE, MODEL_TFLITE,
        MODELS_DIR = 'models_test/'  # model_215
        if not os.path.exists(MODELS_DIR):
            os.mkdir(MODELS_DIR)

        # folder save
        MODEL_TF = MODELS_DIR + f'model_' + target

        # .h5 model
        # MODEL_TF = MODELS_DIR + 'model_' + target + '.h5'

        # .keras model
        # MODEL_TF = MODELS_DIR + 'model_' + target + '.keras'

        # save model after training
        trained_model.model.save(MODEL_TF)

        # save scaler
        # trained_model.model.save(f"model_scaler_saved/{model_name}_model_{target}")
        # # Save the scaler using pickle
        # with open(f"model_scaler_saved/{model_name}_scaler_{target}.pkl", "wb") as scaler_file:
        #     pickle.dump(scaler, scaler_file)

    # make a copy of df
    df_copy = df.copy()

    for number in numbers:
        print(f"You are using {to_save}")
        # when plotting the result, no smooth or sample!!!
        if to_save == "smooth":
            data_prep(df, True, time_step_smooth=number)
        elif to_save == "sample":
            df = data_prep(df, False, sample_number=number)
        else:
            raise KeyError("Method error, smooth or sample?")

        # only care about test errors
        error_no = 5
        # only care about test errors
        error_li = [[] for _ in range(error_no)]
        random_state = [101, 101, 101, 101, 101, 0, 42, 50, 89]

        for i in range(loop_times):
            # print('Dataframe shape after pre-processing->', df.shape)
            print(f"{to_save} {number} experiment->{i + 1}:")

            # define sequence length
            seq_len = 20

            # set the sequence
            ModelEvaluation.set_seq_len(seq_len)

            if target == 'tar1' or target == 'tar2':
                # train test split for one tar
                X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
                    data_split(df=df, tar=target, random_state=random_state[i])
                # create time step for one tar
                X_train_reshaped, X_val_reshaped, X_test_reshaped, y_train_reshaped, y_val_reshaped, y_test_reshaped = \
                    create_time_steps(seq_len, X_train, X_val, X_test, y_train, y_val, y_test)
                # train model for one tar
                trained_model = TrainModel(model_name, X_train_reshaped, X_val_reshaped, input_hyper_params,
                                           y_train_reshaped=y_train_reshaped, y_val_reshaped=y_val_reshaped)
                # save the model for one_tar
                if save_model:
                    save_model_and_scaler(trained_model, model_name, scaler, target)

                # evaluate model for one tar
                model_eva = ModelEvaluation(trained_model.model, X_test_reshaped, target,
                                            y_test_reshaped=y_test_reshaped, y_test=y_test)

            elif target == 'multi_tars':
                # for multi_tars
                X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, scaler = \
                    data_split(df=df, tar=target, random_state=random_state[i])
                # create time step for multi_tars
                X_train_reshaped, X_val_reshaped, X_test_reshaped, y1_train_reshaped, y1_val_reshaped, \
                    y1_test_reshaped = create_time_steps(seq_len, X_train, X_val, X_test, y1_train, y1_val, y1_test)
                _, _, _, y2_train_reshaped, y2_val_reshaped, y2_test_reshaped = \
                    create_time_steps(seq_len, X_train, X_val, X_test, y2_train, y2_val, y2_test)
                # create weight matrix for multi_tars
                weight_matrix_tar1 = create_weight_matrix(len(y1_train_reshaped), input_hyper_params["weight"])
                weight_matrix_tar2 = create_weight_matrix(len(y2_train_reshaped), 1 - input_hyper_params["weight"])
                # train model for multi_tars
                trained_model = TrainModel(model_name, X_train_reshaped, X_val_reshaped, input_hyper_params,
                                           y1_train_reshaped=y1_train_reshaped, y1_val_reshaped=y1_val_reshaped,
                                           y2_train_reshaped=y2_train_reshaped, y2_val_reshaped=y2_val_reshaped,
                                           weight_matrix_tar1=weight_matrix_tar1, weight_matrix_tar2=weight_matrix_tar2,
                                           multi_tars=True)
                # save the model for multi_tars
                if save_model:
                    save_model_and_scaler(trained_model, model_name, scaler, target)

                # evaluate model for multi_tars
                model_eva = ModelEvaluation(trained_model.model, X_test_reshaped, target,
                                            y1_test_reshaped=y1_test_reshaped, y1_test=y1_test,
                                            y2_test_reshaped=y2_test_reshaped, y2_test=y2_test)
            else:
                raise KeyError("The target can only be tar1, tar2, or multi_tars.")

            error_li = record_errors(error_li, model_eva, target)
            print("error list:", error_li)

        # get the original df again
        df = df_copy.copy()

        # write errors into files
        if save_errors:
            save_errors_to_file(error_li, number, to_save, target, input_hyper_params)

