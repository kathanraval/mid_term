import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time
import numpy as np
from pmdarima import auto_arima


# The Model
st.header("K-Mean")


# Business Question on Streamlit
st.subheader("The Question we look to answer is :")


# Reading the Python File
df = pd.read_csv('C:/Users/LENOVO/PycharmProjects/mid_term/Employment_mid_term.csv')


# # Converting the format of the month
# df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)


# Understanding the Plot or the time series
df['New Hires'].plot()
plt.show()


# Converting it into Streamlit
st.subheader("The data looks like:")
st.pyplot(plt.show())


# We nedd to decompse them to make sure that we understand the components and plot thenm
result = seasonal_decompose(df['New Hires'], model='multiplicative', period=12)
result.plot()
plt.show()


# Converting it into Streamlit
st.subheader("The Decompsed Data Looks like:")
st.pyplot(plt.show())


# Fit auto_arima function to AirPassengers dataset to understand which model will be the best out of the 3 ARIMA, SARIMA and SARIMAX
stepwise_fit = auto_arima(df['New Hires'], start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True, d = None, D = 1, trace = True,
                           error_action ='ignore', # we don't want to know if an order does not work
                           suppress_warnings = True, # we don't want convergence warnings
                          stepwise = True)# set to stepwise

# Understanding the summary of the model and which model to fit will be determined by this line of the code
print(stepwise_fit)
print(stepwise_fit.summary())
st.subheader("The Model Option based on the summary")


# Converting it into streamlit
st.write(stepwise_fit.summary())


# Let's split the data in training and testing sets
train = df.iloc[:len(df)-12] # Train data contains 11 out of total 12 years data
test = df.iloc[len(df)-12:] # test data is of 1 year



# Fitting the model
# Remeber that SARIMAX will include SARIMA
model = SARIMAX(train['New Hires'], order = (0, 1, 1), seasonal_order =(2, 1, 1, 12))


# This it the code for the ARIMA with zero seasonalilty this right now is commented
# model = SARIMAX(train['#Passengers'], order = (0, 1, 1), seasonal_order =(0, 0, 0, 0))
result = model.fit()
print(result.summary())


# Covnerting it into Streamlit
st.subheader("Answer of the Chosen Model")
st.write(result.summary())


# predict the values for test data and plot them on one graph
start = len(train)
end = len(train) + len(test) - 1


# Predictions for one-year against the test set
predictions = result.predict(start, end, type='levels').rename("Predictions")


# plot predictions and actual values
predictions.plot(legend=True)
test['New Hires'].plot(legend=True)
plt.show()
st.subheader("The Predicted and Actual ")


# Converting it into Streamlit
st.pyplot(plt.show())


# plot the Errors particularly RMSE
mean_squared_error(test["New Hires"], predictions)
rmseError = rmse(test["New Hires"], predictions)
print(rmseError)


# Converting it into Streamlit
st.subheader("The Root Mean Square Error:")
st.write(rmseError)


# Train the model on the full dataset
model = model = SARIMAX(df['New Hires'], order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))
result = model.fit()


# Forecast for the next 3 years
forecast = result.predict(start=len(df), end=(len(df) - 1) + 3 * 12, type='levels').rename('Forecast')


# Plot the forecast values
df['New Hires'].plot(figsize=(12, 5), legend=True)
forecast.plot(legend=True)
plt.show()


# Converting it into streamlit
st.subheader("The Predicitons plot")
st.pyplot(plt.show())


# Answering the Question on Streamlit
st.subheader("The Answer shows us that")
st.write("")


# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

