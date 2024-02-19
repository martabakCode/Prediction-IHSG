# Impor library yang diperlukan
import numpy as np
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Fungsi untuk memuat dataset dari file CSV
def load_data():
    df = pd.read_csv("dataset.csv")
    # Membersihkan nama kolom dengan menghapus spasi dari awal
    df.columns = list(map(lambda a: a.lstrip(), df.columns))
    
    # Memilih kolom numerik untuk X
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = df.select_dtypes(include=numerics)
    x = df.iloc[:, [1]]
    y = df.iloc[:, -1]
    return df, x, y

# Fungsi untuk melatih model Regresi Pohon Keputusan
def train_model_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    # Membuat model Decision Tree
    tree_model = DecisionTreeRegressor()
    # Melatih model terhadap data
    tree_model = tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    score = tree_model.score(X_test, y_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    return tree_model,score,rms

# Fungsi untuk memprediksi data dengan model Regresi Pohon Keputusan
def predict_decision_tree(x, y, features, day, tanggal):
    tree_model,score,rms = train_model_decision_tree(x, y)
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
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close': closed})
    return predict

# Fungsi untuk melatih model Regresi K-Nearest Neighbors
def train_model_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    knn = KNeighborsRegressor(n_neighbors=2)
    knn = knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    return knn,score,rms

# Fungsi untuk memprediksi data dengan model Regresi K-Nearest Neighbors
def predict_knn(x, y, features, day, tanggal):
    knn,score,rms = train_model_knn(x, y)
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
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close': closed})
    return predict

# Fungsi untuk melatih model Regresi Linier
def train_model_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    linier_model = LinearRegression()
    linier_model = linier_model.fit(X_train, y_train)
    y_pred = linier_model.predict(X_test)
    score = linier_model.score(X_test, y_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    return linier_model,score,rms

# Fungsi untuk memprediksi data dengan model Regresi Linier
def predict_linear_regression(x, y, features, day, tanggal):
    lr,score,rms = train_model_linear_regression(x, y)
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
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close': closed})
    return predict

# Fungsi untuk melatih model Regresi Hutan Acak
def train_model_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)
    rfr = RandomForestRegressor(n_estimators=100)
    rfr_model = rfr.fit(X_train, y_train)
    y_pred = rfr_model.predict(X_test)
    score = rfr_model.score(X_test, y_test)
    rms = np.sqrt(mean_squared_error(y_test, y_pred))
    return rfr_model,score,rms

# Fungsi untuk memprediksi data dengan model Regresi Hutan Acak
def predict_random_forest(x, y, features, day, tanggal):
    rf,score,rms = train_model_random_forest(x, y)
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
    predict = pd.DataFrame({'Date': date, 'Open': opened, 'Close': closed})
    return predict
