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
