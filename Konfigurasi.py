#import library yang dibutuhkan

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os.path
import pandas as pd
import datetime
import altair as alt
import numpy as np
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

#Menu Bar


# Judul dari tab kanan
st.title("Prediksi Data IHSG")

# Menampilkan 5 Data Teratas dan Terbawah
first_row = df.head(5)
separator = pd.DataFrame({'Date': '...','Open' : '...', 'Close':'...'})
last_row = df.tail(5)
result = pd.concat([first_row,separator,last_row])
st.header("Tabel dataset (5 Data Teratas & Terbawah)")
st.table(result)







#membuat sidebar title
st.sidebar.title("Konfigurasi")
st.sidebar.write('Pilih Algoritma yang akan kamu pakai:')
knn = st.sidebar.checkbox('KNN')
lr = st.sidebar.checkbox('Linear Regression')
rf = st.sidebar.checkbox('Random Forest')
dt = st.sidebar.checkbox('Decision Tree')

tanggal = str(df.Date.tail(1))
tanggal = tanggal.split()
tanggal = tanggal[1].split("/")

hari = int(tanggal[0])
bulan = int(tanggal[1])
tahun = int('20'+tanggal[2])
date = st.sidebar.date_input(
    "Tanggal Akhir Prediksi",
    datetime.date(tahun, bulan, hari)+ datetime.timedelta(days=1),
    min_value=datetime.date(tahun, bulan, hari)
)
date_def = datetime.date(tahun, bulan, hari)
day = (date-date_def).days

features = [y.iloc[-1]]
#Tombol Prediksi
button = st.sidebar.button("Prediksi")
tab1, tab2, tab3, tab4 = st.tabs(["KNN", "Linier Regresion","Random Forest","Decision Tree"])
if button:
    if knn:
        with tab1:
            predictionknn = predict_KNN(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil KNN")
            st.table(predictionknn)
    else:
        with tab1:
            st.caption("Untuk menampilkan hasil prediksi menggunakan algoritma KNN silahkan beri tanda check algoritma KNN pada menu konfigurasi")
    if lr:
        with tab2:
            predictionlr = predict_LR(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Linier Regresion")
            st.table(predictionlr)
    else:
        with tab2:
            st.caption("Untuk menampilkan hasil prediksi menggunakan algoritma Linier Regresion silahkan beri tanda check algoritma Linier Regresion pada menu konfigurasi")
    if rf:
        with tab3:
            predictionrf = predict_RF(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Random Forest")
            st.table(predictionrf)
    else:
        with tab3:
            st.caption("Untuk menampilkan hasil prediksi menggunakan algoritma Random Forest silahkan beri tanda check algoritma Random Forest pada menu konfigurasi")
    if dt:
        with tab4:
            predictiondt = predict_DT(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Decision Tree")
            st.table(predictiondt)
    else:
        with tab4:
            st.caption("Untuk menampilkan hasil prediksi menggunakan algoritma Decision Tree silahkan beri tanda check pada menu konfigurasi")

if button:
    st.subheader("Grafik Pembukaan Harga")
    st.caption("Untuk menampilkan grafik prediksi pembukaan harga silahkan beri tanda check pada algoritma KNN, Linier Regresion, Random Forest dan Decision Tree pada menu konfigurasi")
    if dt and rf and lr and knn:
        df['Date'] = pd.to_datetime(df.Date, format='%d/%m/%y').dt.strftime('%Y-%m-%d')
        df['Type'] = "Data Historis"
        predictiondt["Type"] = "Decision Tree"
        predictiondt = [df,predictiondt]
        predictiondt = pd.concat(predictiondt)
        predictiondt['Date'] = pd.to_datetime(predictiondt.Date, format='%Y-%m-%d')
        predictiondt = predictiondt.sort_values(by=['Date'])

        df['Date'] = df.Date
        predictionrf["Type"] = "Random Forest"
        predictionrf = [df,predictionrf]
        predictionrf = pd.concat(predictionrf)
        predictionrf['Date'] = pd.to_datetime(predictionrf.Date, format='%Y-%m-%d')
        predictionrf = predictionrf.sort_values(by=['Date'])

        df['Date'] = df.Date
        predictionlr["Type"] = "Linier Regresion"
        predictionlr = [df,predictionlr]
        predictionlr = pd.concat(predictionlr)
        predictionlr['Date'] = pd.to_datetime(predictionlr.Date, format='%Y-%m-%d')
        predictionlr = predictionlr.sort_values(by=['Date'])

        df['Date'] = df.Date
        predictionknn["Type"] = "KNN"
        predictionknn = [df,predictionknn]
        predictionknn = pd.concat(predictionknn)
        predictionknn['Date'] = pd.to_datetime(predictionknn.Date, format='%Y-%m-%d')
        predictionknn = predictionknn.sort_values(by=['Date'])

        dataclose = pd.concat([predictionrf, predictiondt, predictionlr, predictionknn])
        c = (
        alt.Chart(dataclose)
        .mark_trail()
        .encode(x="Date:T", y="Open", size="Type", color="Type", tooltip=["Date", "Open", "Type"])
        .interactive()
        )

        st.altair_chart(c, use_container_width=True)

