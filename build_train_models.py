import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, \
    GRU, Add, LayerNormalization, BatchNormalization, Layer        # tflite_micro does not support layernorm!!!
from tensorflow.keras import optimizers, Input
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import regularizers
from utils import set_font
tf.get_logger().setLevel('ERROR')


class BuildModels:
    """
    a class for building different nn models -> oneD_cnn, textConvNet, and oneD_cnn_+_GRU
    """

    def __init__(self, model_name: str, hyper_params: dict) -> None:
        self._model_name = model_name
        self._hyper_params = hyper_params

    def print_model_name(self):
        print(f"Using {self._model_name} mdoel...")

    def set_loss(self):
        try:
            if self._hyper_params["loss"] == "Huber":
                loss = tf.keras.losses.Huber(self._hyper_params['delta'])
            elif self._hyper_params["loss"] == "LogCosh":
                loss = tf.keras.losses.LogCosh()
            elif self._hyper_params["loss"] == "MeanAbsoluteError":
                loss = tf.keras.losses.MeanAbsoluteError()
            elif self._hyper_params["loss"] == "MeanSquaredError":
                loss = tf.keras.losses.MeanSquaredError()
            elif self._hyper_params["loss"] == "MeanAbsolutePercentageError":
                loss = tf.keras.losses.MeanAbsolutePercentageError()
            else:
                # if 'loss' is defined but does not match any of the loss errors above
                raise ValueError(f"Invalid loss type: {self._hyper_params['loss']}!"
                                 f"chose from Huber, LogCosh, MeanAbsoluteError, MeanSquaredError,"
                                 f" and MeanAbsolutePercentageError")
        except KeyError:
            # if self.hyper_params["loss"] is not defined
            raise KeyError("Key 'loss' not found in hyper_params")
        return loss

    def set_lr(self):
        """
        set the optimizer, learning rate, return the optimizer
        """
        # set  lr
        optimizer = optimizers.Adam(learning_rate=self._hyper_params['lr'])
        return optimizer

    @staticmethod
    def add_last_layer(model):
        """
        add the last layer, which is a dense layer with linear activation output
        return the model
        """
        # Output Layer with a single neuron for regression
        model.add(Dense(1, activation='linear'))
        return model

    @staticmethod
    def compile_model(model, loss, optimizer) -> None:
        """
        compile the model and return the model
        """
        model.compile(loss=loss, optimizer=optimizer)

    @staticmethod
    def model_summary(model) -> None:
        """
        get the model summary
        """
        model.summary()


class OneDCnn(BuildModels):
        """
        build the model for 1 D CNN
        """
        def __init__(self, model_name: str, X_train_reshaped, hyper_params: dict) -> None:
            super().__init__(model_name, hyper_params)
            self._model_name = model_name
            self._X_train_reshaped = X_train_reshaped

        def output_model(self):
            if self._model_name == "oneD_cnn":
                # print model name
                super().print_model_name()

                # Define the model
                model = Sequential()

                # 1st Convolutional Layer
                model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size1"],
                                 input_shape=self._X_train_reshaped.shape[1:], padding='same'))
                model.add(BatchNormalization())
                model.add(tf.keras.layers.Activation('relu'), )
                # Output shape: batch_size x seq_len x 1 x filters

                # Max Pooling Layer
                model.add(MaxPooling1D(pool_size=2))
                # Output shape: batch_size x seq_len//2 x 1 x filters

                # 2nd Convolutional Layer
                model.add(Conv1D(filters=self._hyper_params["filters2"], kernel_size=self. _hyper_params["kernel_size2"],
                                 input_shape=(10, 1), padding='same'))  # 64
                model.add(BatchNormalization())
                model.add(tf.keras.layers.Activation('relu'))
                # Output shape: batch_size x seq_len//2 x 1 x filters

                # Max Pooling Layer
                model.add(MaxPooling1D(pool_size=2))
                # Output shape: batch_size x seq_len//4 x 1 x filters

                # Flatten the layer
                model.add(Flatten())
                # Output shape: batch_size x seq_len//4 x 1 x filters

                # Fully Connected Layer with 32 neurons
                model.add(Dense(32))
                model.add(BatchNormalization())
                #             model.add(tf.keras.layers.Activation('linear')) # relu for tar1
                model.add(tf.keras.layers.LeakyReLU(alpha=0.01))  # leakyRelu for tar2: 0.1
                # tried LeakyReLu for tar1, a: 0.1 -> loss: 20, a: 0.2 -> loss: 18, a: 0.01 -> loss: 18, a: 0.001 -> loss: 25
                # a: 0.3 -> loss:
                model.add(Dropout(0.3))

                # Fully Connected Layer with 16 neurons
                model.add(Dense(16))
                model.add(BatchNormalization())
                #             model.add(tf.keras.layers.Activation('linear'))  # relu for tar1
                model.add(tf.keras.layers.LeakyReLU(alpha=0.01))  # leakyRelu for tar2
                model.add(Dropout(0.3))

                # Output Layer with a single neuron for regression
                model = BuildModels.add_last_layer(model)

                # set optimizer and lr
                optimizer = super().set_lr()  # 0.001 for tar1, 0.01 for tar2

                # different losses
                loss = super().set_loss()

                # Compile the model
                BuildModels.compile_model(model, loss, optimizer)

                # Display the model summary
                BuildModels.model_summary(model)

                # return the built model
                return model
            else:
                print("This function is for oneD cnn model, check the model name again!")

class TextConvNet(BuildModels):
    """
    build the model TextConvNet
    """

    def __init__(self, model_name: str, X_train_reshaped, hyper_params: dict):
        super().__init__(model_name, hyper_params)
        self._model_name = model_name
        self._X_train_reshaped = X_train_reshaped

    def output_model(self):
        if self._model_name == "text_conv_net":
            # print model name
            super().print_model_name()

            # Define the model
            model = Sequential()

            # 1st Convolutional Layer
            model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size1"],
                             input_shape=self._X_train_reshaped.shape[1:], padding='same'))  # 10
            model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size1"]+1,
                             padding='same'))
            model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            # Output shape: batch_size x seq_len x 1 x filters

            # Max Pooling Layer
            model.add(MaxPooling1D(pool_size=2))
            # Output shape: batch_size x seq_len//2 x 1 x filters

            # 2nd Convolutional Layer
            model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size2"],
                             padding='same'))
            # model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size2"]+1,
                             padding='same'))
            # model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
            # Output shape: batch_size x seq_len//2 x 1 x filters

            # Max Pooling Layer
            model.add(MaxPooling1D(pool_size=2))
            # Output shape: batch_size x seq_len//4 x 1 x filters

            # Flatten the layer
            model.add(Flatten())
            # Output shape: batch_size x seq_len//4 x 1 x filters

            # Fully Connected Layer with 32 neurons
            model.add(Dense(32))
            model.add(BatchNormalization())
            #             model.add(tf.keras.layers.Activation('linear'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))  # alpha: 0.01 for tar1, 0.001 for tar2
            model.add(Dropout(0.3))

            # Fully Connected Layer with 16 neurons
            model.add(Dense(16))
            model.add(BatchNormalization())
            #             model.add(tf.keras.layers.Activation('linear'))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.01))  # alpha: 0.01 for tar1, 0.001 for tar2
            model.add(Dropout(0.3))

            # Output Layer with a single neuron for regression
            model = super().add_last_layer(model)

            # set lr
            optimizer = super().set_lr()  # 0.001 for tar1, 0.01 for tar2

            # different losses
            loss = super().set_loss()

            # Compile the model
            BuildModels.compile_model(model, loss, optimizer)

            # Display the model summary
            BuildModels.model_summary(model)
            return model
        else:
            print("This function is for text conv net model, check the model name again!")

class CnnGru(BuildModels):

    def __init__(self, model_name: str, X_train_reshaped, hyper_params: dict):
        super().__init__(model_name, hyper_params)
        self._model_name = model_name
        self._X_train_reshaped = X_train_reshaped


    def output_model(self):
        '''
        build the model for CNN + GRU
        '''

        if self._model_name == "cnn_gru":
            # print model name
            super().print_model_name()

            model = Sequential()
            # 1st Convolutional Layer
            model.add(Conv1D(filters=self._hyper_params['filters1'], kernel_size=self._hyper_params['kernel_size1'],
                             input_shape=self._X_train_reshaped.shape[1:], padding='same'))
            model.add(BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'), )
            # Output shape: batch_size x seq_len x 1 x filters

            # Max Pooling Layer
            model.add(MaxPooling1D(pool_size=2))
            # Output shape: batch_size x seq_len//2 x 1 x filters

            model.add(GRU(units=self._hyper_params['units'], activation='tanh',
                          input_shape=(self._X_train_reshaped.shape[1] // 2, 1)))

            # Output Layer with a single neuron for regression
            model = super().add_last_layer(model)

            # set optimizer and lr
            optimizer = super().set_lr()

            # different losses
            loss = super().set_loss()

            # Compile the model
            BuildModels.compile_model(model, loss, optimizer)

            # Display the model summary
            BuildModels.model_summary(model)
            return model
        else:
            print("This function is for cnn gru model, check the model name again!")

class TempConvNet(BuildModels):
    dilation_rates = [1, 2, 4, 8]

    def __init__(self, model_name: str, X_train_reshaped, hyper_params: dict, multi_tars=False):
        super().__init__(model_name, hyper_params)
        self._model_name = model_name
        self._X_train_reshaped = X_train_reshaped
        self._multi_tars = multi_tars


    def output_model(self):
        '''
        build the model for TCN
        '''

        if self._model_name == "TCN":
            # print model name
            super().print_model_name()

#             # sequential model
            model = Sequential()

            model.add(Conv1D(filters=self._hyper_params["filters1"], kernel_size=self._hyper_params["kernel_size1"],
                             input_shape=self._X_train_reshaped.shape[1:], padding='same'))

            for dilation_rate in TempConvNet.dilation_rates:
                # x = TempConvNet.residual_block(self, x, dilation_rate)
                model.add(ResidualBlock(filters=self._hyper_params["filters1"],
                                        kernel_size=self._hyper_params["kernel_size1"], dilation_rate=dilation_rate))

            # Flatten layer to prepare for dense layers
            model.add(Flatten())

            # Output layer for one signal neuron
            if not self._multi_tars:
                # outputs = Dense(1, activation='linear')(x)
                # last layer, linear activation function
                model.add(Dense(units=1, activation='linear'))
                # lasy layer, relu activation function
                # model.add(Dense(units=1, activation='relu'))
            elif self._multi_tars:
                # outputs1 = Dense(unit=1, activation='linear', name='tar1')(x)
                # outputs2 = Dense(uint=1, activation='linear', name='tar2')(x)
                model.add(Dense(units=1, activation='linear', name='tar1'))
                model.add(Dense(units=1, activation='linear', name='tar2'))
            else:
                raise ValueError("Check target type again-> tar1, tar2, and multi_tars")


            # set optimizer and lr
            optimizer = super().set_lr()

            # different losses
            loss = super().set_loss()

            # Compile the model
            BuildModels.compile_model(model, loss, optimizer)

            # Display the model summary
            BuildModels.model_summary(model)

            return model

        else:
            print("This function is for TCN model, check the model name again!")


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        # dilated causal conv layers for tar1
        # self.conv1a = Conv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate,
        #                     padding='causal')
        #
        # self.conv1b = Conv1D(filters=self.filters, kernel_size=self.kernel_size, dilation_rate=self.dilation_rate,
        #                     padding='causal')

        # use the causal conv layers without dilation for tar2!!
        self.conv1a = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='causal')
        self.conv1b = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='causal')

        self.norm1 = BatchNormalization()
        self.norm2 = BatchNormalization()

    def call(self, x):
        res = self.conv1a(x)  # x
        res = self.norm1(res)   # x
        res = tf.keras.layers.Activation('relu')(res)  # x
        res = self.conv1b(res)  # x
        res = self.norm2(res)  # x
        res = tf.keras.layers.Activation('relu')(res)  # x
        return Add()([x, res])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "conv1a": self.conv1a,
            "conv1b": self.conv1b,
            "norm1": self.norm1,
            "norm2": self.norm2,
        })
        return config


class MLP(BuildModels):

    def __init__(self, model_name: str, X_train_reshaped, hyper_params: dict):
        super().__init__(model_name, hyper_params)
        self._model_name = model_name
        self._X_train_reshaped = X_train_reshaped

    def output_model(self):
        if self._model_name == "MLP":
            # print model name
            super().print_model_name()

            # sequential API
            # Define the model
            model = Sequential()

            # flatten input
            model.add(Flatten(input_shape=self._X_train_reshaped.shape[1:]))

            # hidden layer1
            model.add(Dense(units=self._hyper_params['n1'], activation='relu'))
            # dropout layer1
            model.add(Dropout(0.3))

            # hidden layer2
            model.add(Dense(units=self._hyper_params['n2'], activation='relu'))
            # dropout layer2
            model.add(Dropout(0.3))


            if self._hyper_params['n3'] != 0:
                # hidden layer3
                model.add(Dense(units=self._hyper_params['n3'], activation='relu'))
                # dropout layer3
                model.add(Dropout(0.3))

            # Output Layer with a single neuron for regression
            model = BuildModels.add_last_layer(model)

            # set optimizer and lr
            optimizer = super().set_lr()  # 0.001 for tar1, 0.01 for tar2

            # different losses
            loss = super().set_loss()

            # Compile the model
            BuildModels.compile_model(model, loss, optimizer)

            # Display the model summary
            BuildModels.model_summary(model)

            # return the built model
            return model

        #     # function API
        #     # input layer
        #     inputs = Input(shape=self._X_train_reshaped.shape[1:])
        #     # flatten layer
        #     x = Flatten()(inputs)
        #     # 1st dense layer
        #     x = Dense(units=self._hyper_params['n1'], activation='relu')(x)
        #     # drop out layers
        #     x = Dropout(0.3)(x)
        #     # 2en dense layer
        #     x = Dense(units=self._hyper_params['n2'], activation='relu')(x)
        #     # drop out layer
        #     x = Dropout(0.3)(x)
        #     # output layer
        #     outputs = Dense(units=1, activation='linear')(x)
        #
        #     # set optimizer and lr
        #     optimizer = super().set_lr()
        #
        #     # create the model
        #     model = Model(inputs, outputs)
        #
        #     # loss function
        #     loss = super().set_loss()
        #
        #     # compile the model
        #     BuildModels.compile_model(model, loss, optimizer)
        #
        #     # Display the model summary
        #     BuildModels.model_summary(model)
        #
        #     return model
        #
        # else:
        #     print("This function is for oneD cnn model, check the model name again!")




class TrainModel:
    def __init__(self, model_name: str, X_train_reshaped, X_val_reshaped, hyper_params,
                 y_train_reshaped=None, y_val_reshaped=None,
                 y1_train_reshaped=None, y1_val_reshaped=None,
                 y2_train_reshaped=None, y2_val_reshaped=None,
                 weight_matrix_tar1=None, weight_matrix_tar2=None,
                 multi_tars=False) -> None:

        self.hyper_params = {
            "filters1": hyper_params["filters1"],
            "kernel_size1": hyper_params["kernel_size1"],
            "loss": hyper_params["loss"],  # Huber, LogCosh,
            "lr": hyper_params["lr"],  # 0.001 for tar 1, 0.01 for tar2
            "delta": hyper_params["delta"]
        }


        # check model name
        if model_name == "oneD_cnn":
            self.hyper_params["filters2"] = hyper_params["filters2"]
            self.hyper_params["kernel_size2"] = hyper_params["kernel_size2"]
            print(self.hyper_params)
            # create an instance from OneDCnn class
            model_init = OneDCnn(model_name, X_train_reshaped, self.hyper_params)
        elif model_name == "cnn_gru":
            self.hyper_params["units"] = hyper_params["units"]
            print(self.hyper_params)
            # create an instance from cnn_gru class
            model_init = CnnGru(model_name, X_train_reshaped, self.hyper_params)
        elif model_name == "text_conv_net":
            self.hyper_params["kernel_size2"] = hyper_params["kernel_size2"]
            print(self.hyper_params)
            # create an instance from TextConvNet class
            model_init = TextConvNet(model_name, X_train_reshaped, self.hyper_params)
        elif model_name == "TCN":
            print(self.hyper_params)
            # create an instance from TCN class
            if not multi_tars:
                # for one tar
                model_init = TempConvNet(model_name, X_train_reshaped, self.hyper_params)
            elif multi_tars:
                # for multi_tars
                model_init = TempConvNet(model_name, X_train_reshaped, self.hyper_params, multi_tars=True)
            else:
                raise ValueError("Check target type again-> tar1, tar2, and multi_tars")

        elif model_name == "MLP":
            self.hyper_params["n1"] = hyper_params["n1"]
            self.hyper_params["n2"] = hyper_params["n2"]
            self.hyper_params["n3"] = hyper_params["n3"]
            print(self.hyper_params)
            # create an instance from MLP class
            model_init = MLP(model_name, X_train_reshaped, self.hyper_params)
        else:
            raise ValueError(f"Invalid model name: {model_name}!")

        self.model = model_init.output_model()

        # define EarlyStopping callback based on testing accuracy
        # original patience is 10
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model with the EarlyStopping callback
        if not multi_tars:
            self.history = self.model.fit(X_train_reshaped, y_train_reshaped,
                                          epochs=70,  # for training 70
                                          batch_size=32,
                                          validation_data=(X_val_reshaped, y_val_reshaped),
                                          callbacks=[early_stopping],
                                          verbose=1)
        elif multi_tars:
            self.history = self.model.fit(x=X_train_reshaped,
                                          y=[y1_train_reshaped,
                                             y2_train_reshaped],
                                          epochs=70,  # for training 70
                                          batch_size=32,
                                          validation_data=(X_val_reshaped,
                                                           [y1_val_reshaped,
                                                            y2_val_reshaped]),
                                          callbacks=[early_stopping],
                                          verbose=1,
                                          sample_weight=[weight_matrix_tar1, weight_matrix_tar2])
        else:
            raise ValueError("Check target type again-> tar1, tar2, and multi_tars")
        # TrainModel.draw_loss(self)


    def draw_loss(self):
        plt.figure(figsize=(8, 4))
        set_font(15, "Times New Roman", 500)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(labels=('Training loss', 'Testing loss'), loc='upper right')
        plt.tight_layout()
        plt.show()



