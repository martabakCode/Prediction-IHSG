# Import library yang dibutuhkan
import streamlit as st
import pandas as pd
import datetime
import altair as alt
from web_functions import load_data, predict_knn, predict_decision_tree, predict_linear_regression, predict_random_forest, train_model_KNN, train_model_DT, train_model_LR, train_model_RF

# Memanggil dataset
df, x, y = load_data()

# Menampilkan judul aplikasi
st.title("Prediksi Data IHSG")

# Menampilkan 5 data teratas dan terbawah dari dataset
first_row = df.head(5)
separator = pd.DataFrame({'Date': ['...'], 'Open': ['...'], 'Close': ['...']})
last_row = df.tail(5)
result = pd.concat([first_row, separator, last_row])
st.header("Tabel dataset (5 Data Teratas & Terbawah)")
st.table(result)

# Membuat sidebar untuk konfigurasi
st.sidebar.title("Konfigurasi")
st.sidebar.write('Pilih Algoritma yang akan kamu pakai:')
algorithms = ['KNN', 'Linear Regression', 'Random Forest', 'Decision Tree']
chosen_algorithms = [st.sidebar.checkbox(algo) for algo in algorithms]


# Memilih tanggal akhir prediksi
tanggal = str(df.Date.tail(1))
tanggal = tanggal.split()
tanggal = tanggal[1].split("/")
hari = int(tanggal[0])
bulan = int(tanggal[1])
tahun = int('20' + tanggal[2])
date = st.sidebar.date_input(
    "Tanggal Akhir Prediksi",
    datetime.date(tahun, bulan, hari) + datetime.timedelta(days=1),
    min_value=datetime.date(tahun, bulan, hari)
)
date_def = datetime.date(tahun, bulan, hari)
day = (date - date_def).days

features = [y.iloc[-1]]

# Tombol untuk prediksi
button = st.sidebar.button("Prediksi")

# Membuat tab untuk hasil prediksi
tabs = st.tabs(["KNN", "Linear Regression", "Random Forest", "Decision Tree"])

if button:
    for algo, tab in zip(algorithms, tabs):
        if chosen_algorithms[algorithms.index(algo)]:
            with tab:
                prediction = globals()[f'predict_{algo.replace(" ", "_")}'.lower()](x, y, [y.iloc[-1]], day,tanggal)
                st.subheader(f"Tabel Hasil {algo}")
                st.table(prediction)
        else:
            with tab:
                st.caption(f"Untuk menampilkan hasil prediksi menggunakan algoritma {algo} silahkan beri tanda check algoritma {algo} pada menu konfigurasi")
    
# Jika tombol prediksi ditekan, tampilkan grafik pembukaan harga
if button:
    st.subheader("Grafik Pembukaan Harga")
    st.caption("Untuk menampilkan grafik prediksi pembukaan harga, beri tanda check pada algoritma KNN, Linier Regresion, Random Forest, dan Decision Tree pada menu konfigurasi.")
    
    selected_predictions = []
    selected_algorithms = []

    for algo in algorithms:
        if chosen_algorithms[algorithms.index(algo)]:
            st.write()
            prediction = globals()[f'predict_{algo.replace(" ", "_")}'.lower()](x, y, [y.iloc[-1]], day, tanggal)
            selected_predictions.append(prediction)
            selected_algorithms.append(algo)

    if selected_predictions:
        data = pd.concat([df.assign(Type="Data Historis")] + [prediction.assign(Type=algo) for algo, prediction in zip(selected_algorithms, selected_predictions)])
        data['Date'] = pd.to_datetime(data.Date, format='%d/%m/%y').dt.strftime('%Y-%m-%d')
        c = alt.Chart(data).mark_trail().encode(x="Date:T", y="Open", size="Type", color="Type", tooltip=["Date", "Open", "Type"]).interactive()
        st.altair_chart(c, use_container_width=True)
        
