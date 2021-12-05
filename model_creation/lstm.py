######
# This file creates a LSTM (Recurrent Neural Network) Model Class
# This class allows you to initialize a LSTM Model and train it to training you feed to it
# You will also be able to use this class to load a stored model and use it for predictions
# Via a button within the app we will train the model
# Via another button we will create predictions with the predict function of the class

import keras
from keras.backend import mean
import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


class LstmModel:
    def __init__(self, df, time_stamp_batches=50) -> None:
        self.df = df
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.time_stamp_batches = time_stamp_batches
        self.predictions = None
        self.model = None
        self.transformer = None
        self.dataset_test = None
        self.predicted_stock_price = None
        self.mae = None
        self.mse = None
        self.prediction_df = None

    def data_preparation(self, trainsize=0.7, col_pos_start=3, col_pos_end=4):
        """
        This function prepares a dataset for training a LSTM Recurrent Neural Network.
        It both splits into a train and test set and uses the MinMaxScaler to normalize the data.

        input: Stock dataframe
        output: Creating Training and testing data within __init__ function of this class
        """

        ######
        # Creating TrainSet
        ######
        training_iloc = round(trainsize * len(self.df))
        df_cols = self.df.columns
        if df_cols[3] == "Close":
            training_set = self.df.iloc[
                :training_iloc, col_pos_start:col_pos_end
            ].values
            test_set = self.df.iloc[training_iloc:, col_pos_start:col_pos_end].values
            print(training_set.shape, test_set.shape)
        else:
            sys.exit(
                "Seems that the `Close`column is not the fourth column of your dataframe. Please either change the order or the parameters col_pos_start and col_pos_end of the function."
            )

        # Lets scale and normalize the data by using the MinMaxScaler
        sc = MinMaxScaler()
        training_set_scaled = sc.fit_transform(training_set)

        # Creating batches of i timestamps and 1 output
        self.X_train = []
        self.y_train = []
        for i in range(self.time_stamp_batches, training_iloc):
            self.X_train.append(training_set_scaled[i - self.time_stamp_batches : i, 0])
            self.y_train.append(training_set_scaled[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_train = np.reshape(
            self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1)
        )

        ######
        # Creating TestSet
        ######
        dataset_train = self.df.iloc[:training_iloc, col_pos_start:col_pos_end]
        self.dataset_test = self.df.iloc[training_iloc:, col_pos_start:col_pos_end]
        dataset_total = pd.concat((dataset_train, self.dataset_test), axis=0)
        inputs = dataset_total[
            len(dataset_total) - len(self.dataset_test) - self.time_stamp_batches :
        ].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)
        self.X_test = []
        for i in range(self.time_stamp_batches, len(inputs)):
            self.X_test.append(inputs[i - self.time_stamp_batches : i, 0])
        self.X_test = np.array(self.X_test)
        self.X_test = np.reshape(
            self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1)
        )

        # Store scaler to __init__ function
        self.transformer = sc

    def model_training(
        self,
        layer_units=50,
        dropout_reg=0.2,
        return_seq=True,
        model_epochs=100,
        model_batch_size=32,
    ):
        self.model = Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        self.model.add(
            LSTM(
                units=layer_units,
                return_sequences=return_seq,
                input_shape=(self.X_train.shape[1], 1),
            )
        )
        self.model.add(Dropout(dropout_reg))
        # Adding a second LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=layer_units, return_sequences=return_seq))
        self.model.add(Dropout(dropout_reg))
        # Adding a third LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=layer_units, return_sequences=return_seq))
        self.model.add(Dropout(dropout_reg))
        # Adding a fourth LSTM layer and some Dropout regularisation
        self.model.add(LSTM(units=layer_units))
        self.model.add(Dropout(dropout_reg))
        # Adding the output layer
        self.model.add(Dense(units=1))

        # Compiling the RNN
        self.model.compile(optimizer="adam", loss="mean_squared_error")

        # Fitting the RNN to the Training set
        self.model.fit(
            self.X_train, self.y_train, epochs=model_epochs, batch_size=model_batch_size
        )

    def model_predicting(self):
        self.predicted_stock_price = self.model.predict(self.X_test)
        self.predicted_stock_price = self.transformer.inverse_transform(
            self.predicted_stock_price
        )
        self.prediction_frame = pd.DataFrame()
        self.prediction_frame["TestData"] = self.dataset_test
        self.prediction_frame["Predictions"] = self.predicted_stock_price

    def model_evaluation(self):
        self.mae = mean_absolute_error(
            self.dataset_test.values, self.predicted_stock_price
        )
        self.mse = mean_squared_error(
            self.dataset_test.values, self.predicted_stock_price
        )

    def model_storing(self):
        pass

    def prediction_evaluation(self):
        pass

    def prediction_storing(self):
        pass
