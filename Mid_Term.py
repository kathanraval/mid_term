import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
st.set_page_config(layout='wide', page_title='Dashboard', page_icon=':sparkles:')
st.title (" Predicitve Analytics Dashboard")



st.write("As a part of our Mid term examination we are asked to create our own Data-set. After making the "
         "Data set we will be using different Machine Learning Techniques on the data set and present 3 "
         "models that are the most appropriate for the data set. Along with 3 models I will be presenting "
         "Data Visualization and Data Summary for the same. ")


st.header("What are different Machine Learning Techniques ?")

st.subheader("Multiple Linear Regression")

st.write("It is a statistical method used in data analysis and machine learning. "
         "In multiple linear regression, the goal is to model the relationship "
         "between a dependent variable (the variable you want to predict) and multiple"
         " independent variables (the variables used to make the prediction). "
         "This method assumes that the relationship between the variables can be expressed as a linear equation.")

st.subheader("Ks Nearest Neighbour")

st.write("K-Nearest Neighbors (KNN) is a simple machine learning algorithm used for classification "
         "and regression tasks. It classifies a data point by considering the majority class of its "
         "K nearest neighbors in the training dataset (for classification) or predicts a numerical "
         "value by averaging the values of its K nearest neighbors (for regression). KNN relies on "
         "distance metrics, like Euclidean distance, to find the closest neighbors")

st.subheader("K-Means")

st.write("K-means is a clustering algorithm used in machine learning and data analysis. "
         "It groups a set of data points into clusters, where each data point belongs to "
         "the cluster with the nearest mean (center). The algorithm iteratively assigns data "
         "points to clusters and updates the cluster centers until convergence, aiming to "
         "minimize the within-cluster variance. K-means is widely used for data segmentation "
         "and pattern recognition.")

st.subheader("Support Vector Machine")

st.write("Support Vector Machine (SVM) is a machine learning algorithm for classification and "
         "regression. It finds the optimal hyperplane that best separates data points into different "
         "classes (for classification) or fits a regression line (for regression) while maximizing "
         "the margin between the classes. SVM works by finding support vectors, which are the data "
         "points closest to the decision boundary, and uses them to define the hyperplane. SVM is"
         " known for its ability to handle high-dimensional data and is effective in various applications")

st.subheader("Time Series")

st.write("Time series refers to a sequence of data points collected or recorded at equally spaced "
         "time intervals. Time series analysis involves examining and modeling such data to make "
         "predictions or uncover patterns over time.")

st.write("ARIMA (AutoRegressive Integrated Moving Average): ARIMA is a widely used time series "
         "forecasting model. It combines auto-regressive (AR), differencing (I), and moving average "
         "(MA) components to model the data. ARIMA models are effective for stationary time series data,"
         " where the mean and variance remain constant over time.")

st.write ("Seasonal ARIMA): SARIMA is an extension of ARIMA that includes seasonal components."
          " It's useful when there are recurring patterns or seasonality in the time series data."
          " SARIMA models include additional terms to capture the seasonal patterns in the data.")

st.write ("Seasonal ARIMA with Exogenous Variables): SARIMAX further extends SARIMA by allowing "
          "for the inclusion of exogenous variables (external factors) in the time series modeling."
          " This enables the model to consider additional factors that may influence the time"
          " series data.")

st.subheader("Principal Component Analysis")

st.write("Principal Component Analysis (PCA) is a dimensionality reduction technique used in data analysis"
         " and machine learning. It simplifies complex datasets by identifying and retaining the most important"
         " information, reducing the number of variables or dimensions while minimizing information loss. PCA "
         "does this by finding new orthogonal axes, called principal components, that capture the largest"
         " variance in the data. These principal components are linear combinations of the original variables,"
         " allowing for a more compact representation of the data. PCA is useful for visualization, noise reduction,"
         " and improving the performance of machine learning algorithms by reducing the dimensionality of the data.")

st.subheader("Logistic Regression")

st.write("Logistic regression is a statistical model used for binary classification tasks in machine learning."
         " It predicts the probability of an event occurring (e.g., yes/no or 1/0) based on one or more predictor"
         " variables. It models the relationship between the predictors and the probability of the event using "
         "the logistic function, which produces an 'S'-shaped curve. This allows logistic regression to estimate "
         "the likelihood of the binary outcome, making it a fundamental tool in classification problems")


st.subheader("The Variables are ")
st.write("Sr.no	Years of Experience	: nmormal income : normal	Salary:normal	No of Previous Employers: normal	Age	: discrete Occupation: unifomr	Job Satisfaction: random 	Education Degree: discrete	Month: time series	New Hires: seasonal	Unemployment Rate: seasonal	")
st.write("the resaons are self evident as they make sense")
st.write("")
st.subheader("Which models I am Using and why")

st.subheader("Time Series")

st.write("To calculate the New Hires")


st.subheader("Naive Bayes")

st.write("TO calculate the job satisfaction")


st.subheader("SVM")

st.write("To calculate the JOb satisfaction")


st.header("How to Run it in your Computer")
st.write("Step 1: Download the Zip file 'mid_term'")
st.write("Step 2: Download the necessary libraries and its dependecies for the code for its smooth running")
st.write("Step 3: Use this code to download the libraries in the CMD: 'pip install _______(library name)'")
st.write("Step 4: The libraries are Streamlit, Pandas, Matplot, Seaborn, numpy, sklearn, statsmodels, pmdarima")
st.write("Step 5: Save the Zip file and open CMD in the same path where the file is saved")
st.write("Step 6: Run either of the following 2 command in the CMD")
st.write("'python -m streamlit run Mid_term.py '")
st.write("'streamlit run Mid_term.py'")

