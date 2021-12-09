import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.indexes.datetimes import date_range
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


class LstmPredictionModel:
    def __init__(self, df, end, backshifting, stock_code):
        self.df = df
        self.end = end
        self.backshifting = backshifting
        self.stock_code = stock_code

    def data_preperation(self):
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
        # Load Test Data
        test_start = self.end
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
        test_plot_df = self.test_df[["Close"]]
        test_plot_df["Prediction"] = self.predicted_prices
        if self.check_delta < 30:
            self.full_test = pd.concat(
                [
                    self.df[["Close"]].iloc[: len(self.df) - self.check_delta],
                    test_plot_df,
                ],
                axis=0,
            )
        else:
            self.full_test = pd.concat([self.df[["Close"]], test_plot_df], axis=0)
        print(self.full_test.dtypes)

    def create_real_prediction(self):
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
