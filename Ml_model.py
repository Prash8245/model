import csv
from pyexpat import model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder  
import webbrowser


df = None
file = st.file_uploader("Upload only csv",['csv'])
if file is not None:
    st.write(file," Uploaded")
    df = pd.read_csv(file)
              
    st.subheader("The First 10 Data \n")
    st.write(df)
    st.subheader("Any Null Values \n")
    st.text(df.isnull().sum())                                             # Check for Null Values
    df.fillna(df.median(),inplace=True)
    columns = df.columns.to_list()
    length = len(columns)
    st.write("Data Frame Size",df.shape)

    values = st.radio('Select x and y index by index or column name',('Index','Column Name'))
    if(values == 'Index'):
        num_x = st.number_input("Enter the x Index",-10,length,1)
        st.write(num_x)
        num_y = st.number_input("Enter the y Index",-10,length,-1)
        x = df.iloc[:,:num_x]
        y = df.iloc[:,num_y]

    else:
        x_columns = st.multiselect("Select The x columns from below",columns)
        y_columns = st.multiselect("Select The y columns from below",columns)
        x = df[x_columns]
        y = df[y_columns]                                                  # Assigning the X and Y values

    if st.button('Label Encoder'):
        lb_coln = st.multiselect('Select the column to encode',(columns))
        lb = LabelEncoder()
        df[lb_coln] = lb.fit_transform(df[lb_coln])

    if st.button('Display'):
        st.write(df.head(10))


    xTrain,xTest,yTrain,yTest = tts(x,y,test_size=0.25)                    # Spliting the Data Into Test And Train Set
    st.write("The xTrain Shape : ",xTrain.shape)
    st.write("The xTest Shape : ",xTest.shape)
    st.write("The yTrain Shape : ",yTrain.shape)
    st.write("The yTest Shape : ",yTest.shape)

classifier = st.sidebar.selectbox("Select the Classifier",('Select','Linear Regression','Logistic Regression','K-NN Model','SVM Model'))


def parameters(classifier):
    param = {}
    if(classifier=='Select'):
        param = None

    elif(classifier=='Logistic Regression' or classifier=='Linear Regression'):
        param = None

    elif(classifier=='K-NN Model'):
        param['K-n'] = st.sidebar.slider("Enter the no of K-Neighbours ",5,200,20)

    elif(classifier=='SVM Model'):
        param['kernel'] = st.sidebar.radio("Select The Kernal",('linear','rbf'))
        if(param['kernel']=='rbf'):
            param['gamma'] = st.sidebar.slider("Select The gamma values ",0.00001,10.0,0.0001)

        param['c'] = st.sidebar.slider("Select the C value",0,50,1)

    return param


def classifiers_model(classifier,param):
    cls = None
    if(classifier=='Select'):
        None

    elif(classifier=='Linear Regression'):
        cls = LinearRegression()

    elif(classifier=='Logistic Regression'):
        cls = LogisticRegression()
        
    elif(classifier=='SVM Model'):
        cls = SVC(kernel=param['kernel'],C=param['c'])

    elif(classifier=='K-NN Model'):
        cls = KNeighborsClassifier(n_neighbors= param['K-n'])

    return cls

param = parameters(classifier)

model = classifiers_model(classifier,param)

if(model==None ):
    pass

elif(classifier=='Linear Regression'):
    model.fit(xTrain,yTrain)
    ypredict = model.predict(xTest)
    acc = model.score(xTest,yTest)
    mod_1 = xTest.copy()
    mod_1['Y actual'] = yTest
    mod_1['Y Predict'] = ypredict
    st.dataframe(mod_1)
    st.subheader("Accuracy of Model")
    st.write("**Accuracy** : ",acc)

else:
    model.fit(xTrain,yTrain)
    ypred = model.predict(xTest)
    acc = model.score(xTest,yTest)

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(yTest,ypred))

    st.subheader("Accuracy of Model")
    st.write("**Accuracy** : ",acc)

    st.subheader("The Classification Report")
    st.write(classification_report(yTest,ypred))

st.spinner('Wait for it...')
#st.balloons()