import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# The Model
st.header("K-Mean")


# Business Question on Streamlit
st.subheader("The Question we look to answer is :")


# Reading the Python File
df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/mid_term/Employment_mid_term.csv")


# Manipulating the Data set, removing the not neede variables
df =df.drop(['Sr.no','income', 'Month', 'New Hires', 'Unemployment Rate',
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
model = SVC(C=15,kernel="rbf",gamma="scale")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(pred)


# Finding the Accuracy Score and Confusion Matrix
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