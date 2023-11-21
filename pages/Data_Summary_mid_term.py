import  seaborn as sns
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/mid_term/Employment_mid_term.csv")

st.header("The Data Visualization :")

df =df.drop(['Sr.no','income', 'Month', 'New Hires', 'Unemployment Rate',
       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'LL',
       'UL', 'Probab', 'Cumalitive', 'Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 12'],axis=1)

print(df.columns)

status = pd.get_dummies(df["Occupation"],drop_first=True)
df = pd.concat([df,status],axis=1)
status = pd.get_dummies(df["Education Degree"],drop_first=True)
df = pd.concat([df,status],axis=1)

print(df.head(10))
print(df.columns)
df = df.drop(['Occupation','Education Degree'],axis=1)

columns_to_standardize = ["Years of Experience","Salary","Age"]
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

summary_stats = df.describe()
mode_values = df.mode()  # For mode
median_values = df.median()  # For Median
variance_values = df.var()  # For Variance
kurtosis_values = df.kurtosis()  # Kurtosis
skewness_values = df.skew()  # Skewness
min = df.min()  # For minimum
max = df.max()  # For Maximum
sum = df.sum()  # For Sum
count = df.count()  # For Count


st.write(summary_stats,mode_values,median_values,variance_values,kurtosis_values,skewness_values,min,max,sum,count)