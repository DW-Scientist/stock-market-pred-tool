# import necessary libraries
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
from model_creation.lstm_model import LstmPredictionModel
import altair as alt
from PIL import Image


# ------------------ Tweek some layout setups of streamlit ----------------- #

# Change Page Favicon to Udacity Logo
fav = Image.open("helpers/udacity_logo.jpeg")
# Set a wide page format which looks better for the kind of charts we show
st.set_page_config(layout="wide", page_icon=fav)
# Set Page Title
st.markdown(
    "<h1 style='text-align: center'>LSTM Stock Prediction Tool</h1>",
    unsafe_allow_html=True,
)

# ------------------ Create the Sidebar of the app to give users some changeable parameters ----------------- #

# 1. Select desired Stock with valid stockcode
stock_option = st.sidebar.text_input(
    "Type int the shortCode of your desired stock like AAPL for Apple", "PYPL"
)

# 2. Select desired Date Range of stock data
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

# 3. Select the training epochs for the LSTM model
selected_training_epochs = st.sidebar.slider(
    "Select the amount of training epochs for the LSTM. Keep in mind the higher the amount of epochs the longer takes the training",
    min_value=25,
    max_value=200,
)
# 4. Select the training batch size for the LSTM model
selected_training_batch_size = st.sidebar.slider(
    "Select the training Batch Size for the LSTM. Keep in mind the lower the size the longer takes the training",
    min_value=32,
)

# 5. Select Backshifting Range - How many days do you use to predict next date
predict_range = st.sidebar.slider(
    "How many past days do you want to use for your prediction?", 0, 90, 30
)


# ------------------ Get Stock Data from yfinance and create ta finance metrics ----------------- #

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

# ------------------ Set up the main part of the App ----------------- #

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

# Create helper functions for downloading a excel file
def excel_creation(df):
    """
    Takes a dataframe and transforms it into an excel file for the download
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, sheet_name="Sheet1")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Create a download link for the excel file
def get_download_link(df):
    """
    Creates a download link for a created excel file
    input: dataframe
    output: href string
    """
    val = excel_creation(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{stock_option}_{select_start_date}_{select_end_date}.xlsx">Download excel file</a>'


# create streamlit element which starts the download if it gets clicked
st.markdown(get_download_link(stock), unsafe_allow_html=True)

# ------------------ Model Training Part ----------------- #

# Creating a title for the user to know which part she/he accesses
st.title("Model Training and Prediction")

# Create button to start TestPredictionPipeline
test_predict_button = st.button(
    f"Train LSTM Model for {stock_option} and create Test Predictions"
)
# Create button to start RealPredictionPipeline
real_predict_button = st.button(
    f"Train LSTM Model for {stock_option} and create Real Predictions"
)

# Start Pipeline if Button gets clicked
if test_predict_button:

    # show gif to let the user know that she/he hast to wait for training to finish
    gif_runner = st.image("helpers/giphy.gif", caption="Training LSTM Model")

    # instantiate LstmPreidctionModel class with required parameters
    lstmmodel = LstmPredictionModel(
        df=stock,
        end=select_end_date,
        backshifting=predict_range,
        stock_code=stock_option,
    )

    # use data_preparation function to transform dataframe into LSTM readible format
    lstmmodel.data_preperation()

    # use model_creation fucntion to create model with sidebar parameters and train it to the given data
    lstmmodel.model_creation(
        input_units=50,
        dropout_rate=0.2,
        epochs=selected_training_epochs,
        batch_size=selected_training_batch_size,
    )

    # use test_predictions function to create some test predictions
    lstmmodel.test_predictions()

    # use test_predictions function to create a dataframe for plotting the data within plotly
    lstmmodel.create_test_prediction_df()

    # stop the gif since processing is over
    gif_runner.empty()

    # plot TestSet vs. Predictions
    st.line_chart(lstmmodel.test_plot_df)
    # plot FullDataset vs. Predictions
    st.line_chart(lstmmodel.full_test)

    # Evaluate the model with its mean absolute and squared error
    lstmmodel.model_evaluation()

    # print mean absolut error
    st.write(
        "The Mean Absolut Error (MAE) of the prediction is `%s` " % (lstmmodel.mae)
    )
    # print mean sqaured error
    st.write(
        "The Mean Squared Error (MSE) of the prediction is `%s` " % (lstmmodel.mse)
    )

# Start Pipeline if Button gets clicked
if real_predict_button:
    # show gif to let the user know that she/he hast to wait for training to finish
    gif_runner = st.image("helpers/giphy.gif", caption="Training LSTM Model")

    # instantiate LstmPreidctionModel class with required parameters
    lstmmodel = LstmPredictionModel(
        df=stock,
        end=select_end_date,
        backshifting=predict_range,
        stock_code=stock_option,
    )

    # use data_preparation function to transform dataframe into LSTM readible format
    lstmmodel.data_preperation()

    # use model_creation fucntion to create model with sidebar parameters and train it to the given data
    lstmmodel.model_creation(
        input_units=50,
        dropout_rate=0.2,
        epochs=selected_training_epochs,
        batch_size=selected_training_batch_size,
    )

    # use test_predictions function to create some test predictions
    lstmmodel.test_predictions()

    # use test_predictions function to create a dataframe for plotting the data within plotly
    lstmmodel.create_test_prediction_df()

    # use the create_real_prediction to predict the next days Closing Price.
    next_day_prediction = lstmmodel.create_real_prediction()
    # Getting the date for the prediction to print it
    prediction_date = list(lstmmodel.test_plot_df.index)[-1] + datetime.timedelta(
        days=1
    )

    # stop the gif since processing is over
    gif_runner.empty()

    # Print the prediction result so the user can see an output
    st.write(
        "The `%s` prediction for the Closing value of the `%s` is: `%s` "
        % (stock_option, prediction_date.date(), next_day_prediction[0][0])
    )
