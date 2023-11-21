import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np
import statsmodels.api as sm
import streamlit as st

# The Model
st.header("K's Nearest Neighbour")


# Business Question on Streamlit
st.subheader("The Question we look to answer is :")


# Reading the Python File
df = pd.read_csv("D:\Kathan\Au Assignment\TOD 310- Predicitive Analytics Business for Business\KNN\iris.csv")
print(df)


# Splitting them into Features and targets (Dependent And Independent Variables)
features = df.drop(["Species","Id"],axis=1)
target = df[["Species"]]


# Splitting the Data into Test and Train
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4)
print(X_train)
print(y_train)
print(X_test)
print(y_test)


# Fitting and Predicting the Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)


# Finding the Accuracy Score and Confusion Matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Setosa",'Versicolor','Verginica'])
disp.plot()
plt.show()
accs = accuracy_score(y_true=y_test,y_pred=y_pred)
print("accuracy score",accs)


# Converting the Confusion Matrix and Accuracy into Streamlit
st.subheader("The Confusion Matrix Looks like")
st.pyplot(plt.show())
st.subheader("The Accuracy Score is")
st.write("Accuracy Score",accs)


# Answering the Question on Streamlit
st.subheader("The Answer shows us that")
st.write("")


# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)
