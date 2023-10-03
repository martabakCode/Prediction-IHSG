#import library yang dibutuhkan

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os.path
import pandas as pd
import datetime
import altair as alt
from sklearn import tree
from csv import writer
from web_functions import load_data
from web_functions import predict_KNN
from web_functions import predict_DT
from web_functions import predict_LR
from web_functions import predict_RF
from web_functions import train_model_KNN
from web_functions import train_model_DT
from web_functions import train_model_LR
from web_functions import train_model_RF

# memanggil dataset
df,x,y = load_data()

# Judul dari tab kanan
st.title("Prediksi Data IHSG")

# Menampilkan 5 Data Teratas dan Terbawah
first_row = df.head(5)
last_row = df.tail(5)
result = pd.concat([first_row,last_row])
st.header("Tabel dataset (5 Data Teratas&Terbawah)")
st.table(result)





#membuat sidebar title
st.sidebar.title("Konfigurasi")
st.sidebar.write('Pilih Algoritma yang akan kamu pakai:')
knn = st.sidebar.checkbox('KNN')
lr = st.sidebar.checkbox('Linear Regression')
rf = st.sidebar.checkbox('Random Forest')
dt = st.sidebar.checkbox('Decision Tree')


date = st.sidebar.date_input(
    "Rentang tanggal Prediksi",
    datetime.date(2023, 8, 29),
    min_value=datetime.date(2023, 8, 29)
)
date_def = datetime.date(2023, 8, 29)
day = (date-date_def).days

features = [y.iloc[-1]]
#Tombol Prediksi
button = st.sidebar.button("Prediksi")
tab1, tab2, tab3, tab4 = st.tabs(["KNN", "Linier Regresion","Random Forest","Decision Tree"])
if button:
    if knn:
        with tab1:
            predictionknn = predict_KNN(x,y,features,day)
            st.subheader("Tabel Hasil KNN")
            st.table(predictionknn)
    if lr:
        with tab2:
            predictionlr = predict_LR(x,y,features,day)
            st.subheader("Tabel Hasil Linier Regresion")
            st.table(predictionlr)
    if rf:
        with tab3:
            predictionrf = predict_RF(x,y,features,day)
            st.subheader("Tabel Hasil Random Forest")
            st.table(predictionrf)
    if dt:
        with tab4:
            predictiondt = predict_DT(x,y,features,day)
            st.subheader("Tabel Hasil Decision Tree")
            st.table(predictiondt)

tab1, tab2 = st.tabs(["Open Grafik", "Close Grafik"])
if dt and rf and lr and knn:
    with tab1:
        
        dataOpen = pd.DataFrame({'Decision Tree': predictiondt.get("Open"), 'Random Forest': predictionrf.get("Open"), 'Linier Regresion': predictionlr.get("Open"), 'KNN': predictionknn.get("Open") })
        st.subheader("Grafik Open")
        st.line_chart(dataOpen)
    with tab2:
        dataClose = pd.DataFrame({'Decision Tree': predictiondt.get("Close"), 'Random Forest': predictionrf.get("Close"), 'Linier Regresion': predictionlr.get("Close"), 'KNN': predictionknn.get("Close") })
        st.subheader("Grafik Close")
        st.line_chart(dataClose)