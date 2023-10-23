import numpy as np
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

def load_data():
    #menambahkan dataset
    df = pd.read_csv("dataset.csv")
    #menghapus semua spasi dari nama kolom
    df.columns = list(map(lambda a: a.lstrip(), df.columns))
    
    #mensett x
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = df.select_dtypes(include=numerics)

    jumlah_column = len(numeric.axes[1])
    
    x = df.iloc[:, [1]]
    y = df.iloc[:, -1]
    return df, x,y

def train_model_DT(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    # membuat model Decision Tree

    tree_model = DecisionTreeRegressor()

    # melakukan pelatihan model terhadap data
    tree_model = tree_model.fit(X_train, y_train)
    
    y_pred = tree_model.predict(X_test)

    return tree_model
    

def predict_DT(x,y,features,day, tanggal):
    tree_model = train_model_DT(x,y)
    hari = int(tanggal[0])
    bulan = int(tanggal[1])
    tahun = int('20'+tanggal[2])

    date = []
    opened = []
    closed = []
    for i in range(day):
        date.append(datetime.date(tahun, bulan, hari) + datetime.timedelta(days=i+1))
        
        if i == 0:
            opened.append(features[0])
            closed.append(tree_model.predict(np.array(features).reshape(1,-1))[0])
        else:
            opened.append(closed[i-1])
            closed.append(tree_model.predict(np.array(closed[i-1]).reshape(1,-1))[0])
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close':closed})
    
    return predict

    
def train_model_KNN(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49)
    
    knn = KNeighborsRegressor(n_neighbors=2)
    
    knn = knn.fit(X_train, y_train)

    return knn

def predict_KNN(x, y, features,day, tanggal):
    knn = train_model_KNN(x, y)

    hari = int(tanggal[0])
    bulan = int(tanggal[1])
    tahun = int('20'+tanggal[2])

    date = []
    opened = []
    closed = []
    for i in range(day):
        date.append(datetime.date(tahun, bulan, hari) + datetime.timedelta(days=i+1))
        
        if i == 0:
            opened.append(features[0])
            
            closed.append(knn.predict(np.array(features,dtype=np.float64).reshape(1,-1))[0])
        else:
            opened.append(closed[i-1])
            closed.append(knn.predict(np.array(closed[i-1],dtype=np.float64).reshape(1,-1))[0])
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close':closed})
    
    return predict

def train_model_LR(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    # membuat model Decision Tree
    linier_model = LinearRegression()

    # melakukan pelatihan model terhadap data
    linier_model = linier_model.fit(X_train, y_train)



    return linier_model

def predict_LR(x, y, features, day, tanggal):
    lr = train_model_LR(x, y)

    hari = int(tanggal[0])
    bulan = int(tanggal[1])
    tahun = int('20'+tanggal[2])

    date = []
    opened = []
    closed = []
    for i in range(day):
        date.append(datetime.date(tahun, bulan, hari) + datetime.timedelta(days=i+1))
        
        if i == 0:
            opened.append(features[0])
            closed.append(lr.predict(np.array(features,dtype=np.float64).reshape(1,-1))[0])
        else:
            opened.append(closed[i-1])
            closed.append(lr.predict(np.array(closed[i-1],dtype=np.float64).reshape(1,-1))[0])
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close':closed})
    return predict

def train_model_RF(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    # membuat model Decision Tree
    rfr = RandomForestRegressor(n_estimators=100)

    # melakukan pelatihan model terhadap data
    rfr_model = rfr.fit(X_train, y_train)



    return rfr_model

def predict_RF(x, y, features, day, tanggal):
    rf = train_model_RF(x, y)

    hari = int(tanggal[0])
    bulan = int(tanggal[1])
    tahun = int('20'+tanggal[2])

    date = []
    opened = []
    closed = []
    for i in range(day):
        date.append(datetime.date(tahun, bulan, hari) + datetime.timedelta(days=i+1))
        
        if i == 0:
            opened.append(features[0])
            closed.append(rf.predict(np.array(features,dtype=np.float64).reshape(1,-1))[0])
        else:
            opened.append(closed[i-1])
            closed.append(rf.predict(np.array(closed[i-1],dtype=np.float64).reshape(1,-1))[0])
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close':closed})
    return predict

hide_st_style = """

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

"""
st.markdown(hide_st_style, unsafe_allow_html=True)