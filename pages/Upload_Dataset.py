# Impor library Streamlit
import streamlit as st
import pathlib
import os.path

# Menampilkan judul halaman
st.markdown("# Upload Dataset")

# Keterangan di bawah judul
st.caption("*Jika anda tidak mengupload dataset baru, maka dataset yang akan digunakan adalah existing dataset (dataset yang telah tersedia), yaitu dataset IHSG yang di ambil pada rentang waktu 2 Maret 2020 - 29 Agustus 2023.")

# Header untuk bagian Upload Dataset
st.header("Dataset Upload")

# Menyediakan opsi untuk mengunggah dataset
uploaded_file = st.file_uploader("Pilih dataset csv dengan format penamaan : dataset.csv", type=['csv'])

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines()
    st.session_state["preview"] = ''
    for i in range(0, min(5, len(data))):
        st.session_state["preview"] += data[i]

# Menampilkan pratinjau CSV
preview = st.text_area("CSV Preview", "", height=150, key="preview")

# Menampilkan pesan upload
upload_state = st.text_area("Pesan Upload", "", key="upload_state")

# Fungsi untuk mengelola proses upload
def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload file terlebih dahulu"
    else:
        data = uploaded_file.getvalue().decode('utf-8')
        parent_path = pathlib.Path(__file__).parent.parent.parent.resolve()
        save_path = os.path.join(parent_path, "prediction-ihsg")
        complete_name = os.path.join(save_path, uploaded_file.name)
        destination_file = open(complete_name, "w")
        destination_file.write(data)
        destination_file.close()
        st.session_state["upload_state"] = "Penyimpanan " + complete_name + " berhasil!"

# Tombol untuk memicu fungsi upload
st.button("Upload dataset", on_click=upload)
