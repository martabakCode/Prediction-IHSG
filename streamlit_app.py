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

tanggal = str(df.Date.tail(1))
tanggal = tanggal.split()
tanggal = tanggal[1].split("/")

hari = int(tanggal[0])
bulan = int(tanggal[1])
tahun = int('20'+tanggal[2])
date = st.sidebar.date_input(
    "Rentang tanggal Prediksi",
    datetime.date(tahun, bulan, hari),
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
    if lr:
        with tab2:
            predictionlr = predict_LR(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Linier Regresion")
            st.table(predictionlr)
    if rf:
        with tab3:
            predictionrf = predict_RF(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Random Forest")
            st.table(predictionrf)
    if dt:
        with tab4:
            predictiondt = predict_DT(x,y,features,day,tanggal)
            st.subheader("Tabel Hasil Decision Tree")
            st.table(predictiondt)

if button:
    if dt and rf and lr and knn:
        st.subheader("Grafik Open")
        df['Date'] = pd.to_datetime(df.Date, format='%d/%m/%y').dt.strftime('%Y-%m-%d')
        df['Type'] = "Data"
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

# File Upload
st.header("File Upload")
uploaded_file = st.file_uploader("Choose a CSV file name : dataset.csv")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines()         
    st.session_state["preview"] = ''
    for i in range(0, min(5, len(data))):
        st.session_state["preview"] += data[i]
preview = st.text_area("CSV Preview", "", height=150, key="preview")
upload_state = st.text_area("Upload State", "", key="upload_state")

def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data = uploaded_file.getvalue().decode('utf-8')
        parent_path = pathlib.Path(__file__).parent.parent.resolve()           
        save_path = os.path.join(parent_path, "prediction")
        complete_name = os.path.join(save_path, uploaded_file.name)
        destination_file = open(complete_name, "w")
        destination_file.write(data)
        destination_file.close()
        st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
st.button("Upload file to Sandbox", on_click=upload)