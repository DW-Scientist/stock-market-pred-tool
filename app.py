import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import base64
import time
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from io import BytesIO
from model_creation.lstm import LstmModel
from model_creation.lstm_model import LstmPredictionModel
import altair as alt


# Create streamlit application
# Set a wide page format which looks better for the kind of charts we show
st.set_page_config(layout="wide")

# Set Page Title
st.markdown(
    "<h1 style='text-align: center'>LSTM Stock Prediction Tool</h1>",
    unsafe_allow_html=True,
)


########
# sidebar #
########

# 1. Select desired Stock
stock_option = st.sidebar.text_input(
    "Type int the shortCode of your desired stock like AAPL for Apple", "AAPL"
)

# 2. Select desired Date Range
today = datetime.date.today()
default_start = today - datetime.timedelta(days=365)
select_start_date = st.sidebar.date_input("Select Start Date", default_start)
select_end_date = st.sidebar.date_input("Select End Date", today)

# check for right date selection
if select_start_date < select_end_date:
    st.sidebar.success(
        "Start Date: `%s`\n\nEnd Date: `%s`" % (select_start_date, select_end_date)
    )
else:
    st.sidebar.error("Your Start Date has to be smaller than your End Date")


# Prediction Part
## Select Range of Days to predict
predict_range = st.sidebar.slider(
    "How many days into the future do you want to predict?", 0, 90, 30
)

## OtherStuff to come
# first_prediction_date = select_end_date + datetime.timedelta(days=1)
# date_index_range = pd.date_range(first_prediction_date, periods=predict_range, freq="D")
# predict_template = pd.DataFrame(
#     index=date_index_range,
#     columns=[
#         "Open",
#         "High",
#         "Low",
#         "Close",
#         "Adj Close",
#         "Volume",
#     ],
# ).reset_index()
# predict_template.columns = [
#     "Date",
#     "Open",
#     "High",
#     "Low",
#     "Close",
#     "Adj Close",
#     "Volume",
# ]
# for col in predict_template.columns[1:]:
#     predict_template[col] = 0
# predict_template["Date"] = predict_template["Date"].dt.date
# predict_template.set_index("Date", inplace=True)


########
# Get Stock Data from yfinance #
########

# Call yfinance API and test if user stock input is a valid entry
try:
    stock = yf.download(
        stock_option, start=select_start_date, end=select_end_date, progress=False
    )
    if len(stock) > 0:
        # some evaluation metrics
        # Bollinger Bands
        bollinger = BollingerBands(stock.Close)
        bollinger_df = stock.copy()
        bollinger_df["bol_hband"] = bollinger.bollinger_hband()
        bollinger_df["bol_lband"] = bollinger.bollinger_lband()
        bollinger_df = bollinger_df[["Close", "bol_hband", "bol_lband"]]

        # Moving Average Convergence Divergence
        macd = MACD(stock.Close).macd()

        # Resistence Strength Indicator
        rsi = RSIIndicator(stock.Close).rsi()

    else:
        st.sidebar.error(
            "You didn't chose a valid StockCode. Try to find sth like AAPL"
        )
except:
    st.sidebar.error("You didn't chose a valid StockCode. Try to find sth like AAPL")

########
# Set up the main part of the App
########

# Create the Plot with the bollinger metric
st.write("`%s` Stock and its Bollinger Band Metric" % (stock_option))
st.line_chart(bollinger_df)

# Plot the MACD metric
st.write("`%s` Stock Moving Average Convergence Divergence (MACD)" % (stock_option))
st.area_chart(macd)

# Plot the RSI Metric
st.write("`%s` Stock Relative Strength Index (RSI)" % (stock_option))
st.line_chart(rsi)

# Create Part for downloading data
st.write(f"{stock_option} Data To Download (Preview last 10 Days)")
st.dataframe(stock.tail(10))

# Create helper functions for downloading a csv file
def excel_creation(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_download_link(df):
    """
    Creates a download link for a created excel file
    input: dataframe
    output: href string
    """
    val = excel_creation(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{stock_option}_{select_start_date}_{select_end_date}.xlsx">Download excel file</a>'


st.markdown(get_download_link(stock), unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    test_predict_button = st.button(
        f"Train LSTM Model for {stock_option} and create Test Predictions"
    )
with right:
    real_predict_button = st.button(
        f"Train LSTM Model for {stock_option} and create Real Predictions"
    )

# start_training = st.button(
#     f"Train LSTM Model for {stock_option} and create Test Predictions"
# )
if test_predict_button:
    gif_runner = st.image("data/helpers/giphy.gif", caption="Training LSTM Model")

    #####
    # instantiate the lstm class and train the model
    #####
    lstm_stock = LstmModel(df=stock)

    # prepare dataset
    lstm_stock.data_preparation()

    # create and train the model
    lstm_stock.model_training()

    gif_runner.empty()

    # create predictions
    lstm_stock.model_predicting()

    # create evaluation metrics
    lstm_stock.model_evaluation()

    # plot test prediction and print evaluation metrics
    st.line_chart(lstm_stock.prediction_frame[["TestData", "Predictions"]])
    st.write(
        "The Mean Absolut Error (MAE) of the prediction is `%s` " % (lstm_stock.mae)
    )
    st.write(
        "The Mean Squared Error (MSE) of the prediction is `%s` " % (lstm_stock.mse)
    )


if real_predict_button:
    gif_runner = st.image("data/helpers/giphy.gif", caption="Training LSTM Model")

    # #####
    # # instantiate the lstm class and train the model
    # #####
    # lstm_stock = LstmModel(
    #     df=stock,
    #     test_prediction=True,
    #     test=True,
    #     prediction_start=select_end_date,
    #     prediction_range=predict_range,
    # )

    # # prepare dataset
    # lstm_stock.data_preparation()

    # # create and train the model
    # lstm_stock.model_training()

    # gif_runner.empty()

    # # create predictions
    # lstm_stock.model_predicting()

    # # plot test prediction and print evaluation metrics
    # st.line_chart(lstm_stock.prediction_frame[["TrainData", "Predictions"]])

    lstmmodel = LstmPredictionModel(
        df=stock,
        end=select_end_date,
        backshifting=predict_range,
        stock_code=stock_option,
    )

    lstmmodel.data_preperation()

    lstmmodel.model_creation(input_units=50, dropout_rate=0.2, epochs=25, batch_size=32)

    lstmmodel.test_predictions()

    lstmmodel.create_test_prediction_df()

    lstmmodel.create_real_prediction()

    st.line_chart(lstmmodel.full_test)

    gif_runner.empty()
