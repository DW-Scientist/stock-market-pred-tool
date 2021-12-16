# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.datetimes import date_range
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Creation our LSTM Model Class which will include functions for our button pipeline logic in the app
class LstmPredictionModel:
    """
    This Class has been built for creating stockmarket predictions. It contains different functions which enable a from start to end training pipeline.
    The functions store different variables into the __init__ function so they'll be accessable within the different steps of the app.
    Required Parameters:
    df: type(pandas dataframe): Pandas Dataframe wich contains stock data from yfinance library
    end: type(datetime): End date of the users dateselection
    backshifting: type(integer): Declares how many past days the model uses to predict the next day
    stock_code:  type(string): Necessary to load the right data from yfinance

    """

    def __init__(self, df, end, backshifting, stock_code):
        self.df = df
        self.end = end
        self.backshifting = backshifting
        self.stock_code = stock_code

    def data_preperation(self):
        """
        This function scales the data to normalize it and creates a X_train and y_train set.
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.df["Close"].values.reshape(-1, 1))

        self.X_train = []
        self.y_train = []

        for x in range(self.backshifting, len(scaled_data)):
            self.X_train.append(scaled_data[x - self.backshifting : x, 0])
            self.y_train.append(scaled_data[x, 0])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_train = np.reshape(
            self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1)
        )

    def model_creation(self, input_units, dropout_rate, epochs, batch_size):
        """
        This function creates a LSTM model by using the input parameters of the function. It will also already train the model to the training data.
        """
        ## Build Model
        self.model = Sequential()

        self.model.add(
            LSTM(
                units=input_units,
                return_sequences=True,
                input_shape=(self.X_train.shape[1], 1),
            )
        )
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=input_units, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=input_units))
        self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(units=1))  # Prediction of next closing value

        self.model.compile(optimizer="adam", loss="mean_absolute_error")

        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def test_predictions(self):
        """
        This function creates a Test set which enables the model to create some test predictions.
        It also rescales the data so to have it back in the right format.
        """
        # Load Test Data
        test_start = self.end + dt.timedelta(days=1)
        test_end = dt.datetime.now().date()
        self.check_delta = (test_end - test_start).days
        if self.check_delta < 30:
            test_start = test_start - dt.timedelta(days=30)

        self.test_df = yf.download(
            self.stock_code, start=test_start, end=test_end, progress=False
        )

        ## actual test prices
        self.test_prices = self.test_df["Close"].values

        self.total = pd.concat((self.df["Close"], self.test_df["Close"]), axis=0)

        self.model_inputs = self.total[
            len(self.total) - len(self.test_df) - self.backshifting :
        ].values
        self.model_inputs = self.model_inputs.reshape(-1, 1)
        self.model_inputs = self.scaler.transform(self.model_inputs)

        ## Make Predictions on Test Data

        self.X_test = []

        for x in range(self.backshifting, len(self.model_inputs)):
            self.X_test.append(self.model_inputs[x - self.backshifting : x, 0])

        self.X_test = np.array(self.X_test)
        self.X_test = np.reshape(
            self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1)
        )

        self.predicted_prices = self.model.predict(self.X_test)
        self.predicted_prices = self.scaler.inverse_transform(self.predicted_prices)

    def create_test_prediction_df(self):
        """
        This function creates dataframes which can be used for plotting the test results.
        There is a frame for both only testdata and predictions and also for the whole data.
        """
        self.test_plot_df = self.test_df[["Close"]].copy()
        self.test_plot_df["Prediction"] = self.predicted_prices
        if self.check_delta < 30:
            self.full_test = (
                self.df[["Close"]]
                .iloc[: len(self.df) - self.check_delta]
                .join(self.test_plot_df[["Prediction"]])
            )
        else:
            self.full_test = pd.concat(
                [self.df[["Close"]], self.test_plot_df[["Prediction"]]], axis=0
            )

    def create_real_prediction(self):
        """
        This function creates a prediction for the day after the last day of the test dataset.
        """
        real_data = [
            self.model_inputs[
                len(self.model_inputs) - self.backshifting : len(self.model_inputs), 0
            ]
        ]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = self.model.predict(real_data)
        prediction = self.scaler.inverse_transform(prediction)
        return prediction

    def model_evaluation(self):
        """
        This function creates both the mean absolute error and the mean squared error for the test prediction.
        """
        self.mae = mean_absolute_error(
            self.test_plot_df["Close"], self.test_plot_df["Prediction"]
        )
        self.mse = mean_squared_error(
            self.test_plot_df["Close"], self.test_plot_df["Prediction"]
        )
