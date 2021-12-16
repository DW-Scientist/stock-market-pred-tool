# stock-market-pred-tool

Since I am really interested into the stock market, I want to find how LSTM Networks perform when it comes to predicting the Closing price of a specific stock. That is why I have chosen the following project as my capstone project for the Udacity Data Science Nanodegree: <br>
<p align="center">
  <strong>---- Buidling a Stock Market Ananlyzing/Prediciton App ----</strong>
</p>

## Approach and Libraries 
There are librarys I should mention since they are really important for the project: 
- **Yahoo Finance:** I used this library to query the stock market data from yahoo finance platform 
- **Stremalit:** Streamlit is a library which allows python / data people to quickle set up analytics and Data Science apps without focusing on Fronted languages like HTML, CSS or JS 
- **Keras / Tensorflow:** I used those library to be able to train a LSTM Network (Long Short-Term Memory)

## Set up yourself up ang get started locally
Please follow the upcoming steps to set up the app on your local machine:
1. Clone this repository into your desired location with the command `git clone`
2. `cd`into the repository and create a `venv` with the command `python3 -m venv venv`. This creates a virtual environment with the name `venv`
3. Now you have to activate your environment the the command `source venv/bin/activate`
4. Now it is activated you have to install alle the required packages. For this we use the `requirements.txt` file. To do that type `pip install -r requirements.txt`
5. If you didn't get any errors you should now be able to run the app. Type the command `streamlit run app.py`. It should automatically open a browser window. If not, open the address which gets showed in your terminal.

## The App and its functionalities
If you've made it through the setup up steps you should see sth like the following:
<p align="center">
  <img width="1424" alt="Bildschirmfoto 2021-12-16 um 12 30 04" src="https://user-images.githubusercontent.com/65920261/146364003-d35e4db8-a1e0-48f7-b5c4-a2383d0f9a68.png">
</p>
By default the app loads stock data for the company Paypal for one year. Both the data and some stock market metrics get directly displayed on the right wichtin some visuals. But what can you see there exactly: 
- The first chart shows both the raw stock data and the bollinger bands. Based on a 20-days moving average the standard deviation of every point gets added for the higher boollinger band and subtracted for the lower bolling band. You can read up more <a href="https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/bollinger-bands#:~:text=Bollinger%20Bands%20are%20envelopes%20plotted,moving%20average%20of%20the%20price.&text=Bollinger%20bands%20help%20determine%20whether,conjunction%20with%20a%20moving%20average">here</a>:
<p align="center">
  <img width="1074" alt="Bildschirmfoto 2021-12-16 um 12 34 32" src="https://user-images.githubusercontent.com/65920261/146369539-3e43ff0d-07a9-41bc-aae6-bdf70446b8c8.png">
</p>
- The second chart shows the MACD (Moving Average Convergence-Divergence) of the data. This curve is built based on different exponential moving averages. When the curve is crossing the axis this is either a buy signal (up) or a sell signal (down). Read up more <a href="https://www.investopedia.com/terms/m/macd.asp">here</a>:
<p align="center">
  <img width="1063" alt="Bildschirmfoto 2021-12-16 um 12 35 00" src="https://user-images.githubusercontent.com/65920261/146369644-438461a4-6e4d-4d3a-834f-43cbc252f989.png">
</p>
- The third chart shows the RSI (Relative Strength Index) of the data. This metric is helpful to identify short-term Highs or Lows. It moves between 0 and 100. Signals can be set for instance at 30 (low) and 70 (high). If the price rises from below to over the 30 mark you get a buy signal. If the price falls from over to below 70 you would get a sell signal. You can find out more <a href="https://www.investopedia.com/terms/r/rsi.asp">here</a>:
<p align="center">
<img width="1070" alt="Bildschirmfoto 2021-12-16 um 12 35 07" src="https://user-images.githubusercontent.com/65920261/146369733-2bb82f33-ad28-4a81-9523-9759a565f680.png">
</p>
- Below the visuals you can see a preview of the raw stock data as a dataframe. Below the table you can click `Download excel file` to download the stock data as an Excel file. 
<p align="center">
<img width="663" alt="Bildschirmfoto 2021-12-16 um 12 35 17" src="https://user-images.githubusercontent.com/65920261/146380809-93cd101a-ebb2-4567-bdee-131e6e9aca8c.png">
</p>

## Sidebar and its parameters 
Streamlit gives a great oppurtinity to allow the user to set her/his own filters. In the following I will describe you which parameters you can adjust and how this will affect the solution of the app:
- Data Analytics Parameters
<p align="center">
  <img width="252" alt="Bildschirmfoto 2021-12-16 um 14 33 09" src="https://user-images.githubusercontent.com/65920261/146383308-c7318bcf-5e3b-4b28-9dc8-c288394369a7.png">
</p>
Those parameters are responsible for the stock data you want to query from yfinance. First you have to enter a valid Stock code - The default is **PYPL** for Paypal. You could also use sth like AAPL or TESL. You can just google for those codes. Within the datepickers below you can specify the time range which will be the base for your stock query. The default is 1 year of data backshifted from today on.
- Data Science parameters
<p align="center">
<img width="253" alt="Bildschirmfoto 2021-12-16 um 14 33 24" src="https://user-images.githubusercontent.com/65920261/146383979-1439640a-e8c1-4215-85cd-d7b2c24c8c2f.png">
</p>
Those paramters affect the training process of the underlying LSTM models class. First of all you can change the number of training epochs. Increasing the number of epochs can enhance your result but also leads to a longer training time. The default value is 25. The second parameter you can tweek is the training batch size. Keep in mind the lower this size is the longer your training time since you batch the data into smaller pieces. The third parameter is responsible for training backshifting. That means how many days into the past do you want to take into account to predict a specific data point. The default is 30 - so you look at the last 30 days to predict the upcoming day. 

## Training, Testing and Predicting
Now to the interesting part. At the bottom of the page you should see the following:
<p align="center">
<img width="1029" alt="Bildschirmfoto 2021-12-16 um 15 31 37" src="https://user-images.githubusercontent.com/65920261/146390803-f6561a77-d0a8-4f2f-a8ca-42dc86beebd5.png">
</p>
The two different buttons you see have different functionalities:
- **Train LSTM Model and create Test Predictions**
By clicking this button the model takes the data you've selected above and trains model on it which creates some Test Predictions. The Predictions are based on the last part of your dataset. After you've clicked the button you will see a rocket gif which means that your model is training. For this uses the training parameters on the sidebar mentioned above. When the training process is finished you will see the following visuals and some evaluation metrics which help you to see how well the model performed:
<p align="center">
<img width="1077" alt="Bildschirmfoto 2021-12-16 um 15 42 22" src="https://user-images.githubusercontent.com/65920261/146392711-05f00252-70d0-4599-9771-9b04a31273df.png">
</p>
You can try to tweek the models parameters to get a better result.
- **Train LSTM Model for `DesiredStockCode` and create Real Predictions**
The second button goes through the same process as the first one. But this time the output will be a prediction of the Closing price for the upcoming day. **Please note that the model predicts the Closing price based on your selected date range. So if you want to predict the next days Closing price just leave the Datepicker for the End Date on top to its default which is the current date**. The output would look like the following:
<p align="center">
<img width="969" alt="Bildschirmfoto 2021-12-16 um 15 45 56" src="https://user-images.githubusercontent.com/65920261/146393174-04e65669-8a91-4258-a507-823c5146ebb9.png">
</p>

## Last words
Please do not consider the result of the model as the real upcoming stock price prediction. So please do not use this as a buying/selling advertising tool. This project was supposed to enhance some data science and computer science skills. <br>
Feel free to play around with the code and start creating your own stock-analyzing app. <br>
Thx a lot to Udacity for the great Nanodegree program!!
