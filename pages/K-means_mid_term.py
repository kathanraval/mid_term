import pandas as pd
import statsmodels.compat
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import streamlit as st

# The Model
st.header("K-Mean")


# Business Question on Streamlit
st.subheader("The Question we look to answer is :")


# Reading the Python File
df = pd.read_csv("C:/Users/LENOVO/PycharmProjects/mid_term/Employment_mid_term.csv")
print(df)


# Manipulating the Data set, removing the not neede variables
df =df.drop(['Sr.no','Month', 'New Hires', 'Unemployment Rate',
       'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'LL',
       'UL', 'Probab', 'Cumalitive', 'Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 12'],axis=1)


# Getting Dummies, Converting Catergorical variables into interger variables
# status = pd.get_dummies(df["Occupation"],drop_first=True)
# df = pd.concat([df,status],axis=1)
status = pd.get_dummies(df["Education Degree"],drop_first=True)
df = pd.concat([df,status],axis=1)


# Reshaping the Data Set, dropping categorical variables column names
df = df.drop(['Education Degree'],axis=1)


# Splitting them into Features and targets (Dependent And Independent Variables)
features = df.loc[:, ["Years of Experience","Salary","Masters","PHD","income","No of Previous Employers"]]
target = df.loc[:, "Occupation"]
#
#
# Converting categorical values into the integer values
le = LabelEncoder()
target = le.fit_transform(target)



# Finding the Number of clusters required for the clustering
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)


# Plotting the Clusters
fig, ax = plt.subplots()
ax.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.figure(figsize=(5, 3))
plt.show()


# Converting them into Streamlit
st.pyplot(fig)


# Splitting the Data into Test and Train
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)


# Fitting and Predicting the Model
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(features)
fig, ax = plt.subplots(figsize=(5,3))
ax.scatter(df['No of Previous Employers'], df['Age'], c=kmeans.labels_)
plt.show()


# Converting it into Streamlit
st.pyplot(plt.show())


# Finding the Accuracy Score and Confusion Matrix
pred = kmeans.predict(X_test)
accs = accuracy_score(y_test, pred)
print("Accuracy Score:", accs)
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Doctor','Engineer','Professor'])
disp.plot()
plt.show()


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