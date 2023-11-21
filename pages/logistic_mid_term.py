import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn import metrics
from sklearn.metrics import roc_auc_score

# The Model
st.header("Logistic Regression")


# Business Question on Streamlit
st.subheader("The Question we look to answer is :")


# Reading the Python File
df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/mid_term/Employment_mid_term.csv")



# Manipulating the Data set, removing the not neede variables
df =df.drop(['Sr.no','income ', 'Month', 'New Hires', 'Unemployment Rate',
       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'LL',
       'UL', 'Probab', 'Cumalitive', 'Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 12'],axis=1)


# Getting Dummies, Converting Catergorical variables into interger variables
status = pd.get_dummies(df["Occupation"],drop_first=True)
df = pd.concat([df,status],axis=1)
status = pd.get_dummies(df["Education Degree"],drop_first=True)
df = pd.concat([df,status],axis=1)


# Reshaping the Data Set, dropping categorical variables column names
df = df.drop(['Occupation','Education Degree'],axis=1)


# Standardizing the data
columns_to_standardize = ["Years of Experience","Salary","Age"]
scaler = StandardScaler()
df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])


# Splitting them into Features and targets (Dependent And Independent Variables)
feautres = df.loc[:, ["Years of Experience","Salary","Engineer","Professor","Masters","PHD"]]
target = df.loc[:, "Job Satisfaction"]


# Splitting the Data into Test and Train
X_train, X_test, y_train, y_test = train_test_split(feautres, target, test_size=0.2, random_state=4)


# Fitting and Predicting the Model
model = LogisticRegression(C=1, random_state=4)
model.fit(X_train, y_train)
pred = model.predict(X_test)


# Producing the Root Mean square error and the "S" Shape Curve
rmse = metrics.mean_squared_error(y_test, pred) ** 0.5
fpr, tpr, thresholds = metrics.roc_curve(y_test, model.decision_function(X_test))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('Sensitivity')
plt.ylabel('Specificity')


# Converting it into Streamlit
st.subheader("The Root Mean Square Error is")
st.write(rmse)
st.subheader("The 'S' Shaped Curve looks like")
st.pyplot(plt.show())


# Confusion Matrix, Confusion Matrix Display and Accuracy Score
accs = accuracy_score(y_test,y_pred=pred)
print(accs)
cm = confusion_matrix(y_test,y_pred=pred)
print(cm)
disp = ConfusionMatrixDisplay(cm, display_labels=['1','0'])
disp.plot()
plt.show()


# For Converting this into Streamlit
st.subheader("The Confusion Matrix Looks like:")
st.pyplot(disp.plot().figure_)
st.subheader("The Accuracy Score is:")
st.write("Accuracy Score:", accs)


# Answering the Question on Streamlit
st.subheader("The Answer shows us that")
st.write("")


# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)
