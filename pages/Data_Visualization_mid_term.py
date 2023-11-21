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


# For correlation Matrix and Heat Map
correlation_matrix = df.corr()
print(correlation_matrix)
plt.figure(figsize=(5, 3))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
st.subheader("Correlation Matrix")
st.pyplot()

# For pairplot
sns.pairplot(df)
plt.show()
st.subheader("Pairplot")
st.pyplot()

# For Histogram
df.hist(figsize=(12, 8))
plt.show()
st.subheader("Histogram")
st.pyplot()

# For Box Plot
sns.boxplot(data=df)
plt.show()
st.subheader("Box Plot")
st.pyplot()

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)