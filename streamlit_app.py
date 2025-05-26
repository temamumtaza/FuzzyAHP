#Import library yang dibutuhkan
import pandas as pd 
import numpy as np 
import base64
import xlsxwriter
from io import BytesIO
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FAHP : Seleksi Keringanan UKT", layout="wide",menu_items=None)

#Fungsi untuk membaca file dan menyimpan dalam bentuk array / tuple 
def read_excel_file(filename, n):
    df = pd.read_excel(filename)
    items = np.array(df.iloc[:, 0].tolist()) if n == 0 else tuple(zip(df.iloc[:, 0].tolist(), df.iloc[:, n].tolist()))
    return items

# def filedownload(df, filename='output.xlsx'):
#     output = BytesIO()
#     writer = pd.ExcelWriter(output, engine='xlsxwriter') # ubah engine ke xlsxwriter
#     df.to_excel(writer, index=False, sheet_name='Sheet1')
#     writer.save()  # Menggunakan writer.save() untuk menyimpan workbook
#     processed_data = output.getvalue()
#     b64 = base64.b64encode(processed_data)
#     return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download file</a>'

def isConsistent(matrix, printComp=True):
    mat_len = len(matrix)
    RI = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    midMatrix = np.array([m[1] for row in matrix for m in row]).reshape(mat_len, mat_len)
    if(printComp): st.write("mid-value matrix: \n", midMatrix, "\n")
    
    eigenvalue = np.linalg.eigvals(midMatrix)
    lambdaMax = max(eigenvalue)
    if(printComp): st.write("eigenvalue: ", eigenvalue)
    if(printComp): st.write("lambdaMax: ", lambdaMax)
    if(printComp): st.write("\n")

    if mat_len >= 10:
        RIValue = RI[10]
    else:
        RIValue = (lambdaMax - mat_len)/(mat_len-1)
    st.write("R.I. Value: ", RIValue)

    CIValue = (lambdaMax-mat_len)/(mat_len - 1)
    st.write("C.I. Value: ", CIValue)

    CRValue = CIValue/RIValue
    st.write("C.R. Value: ", CRValue)

    if(printComp): st.write("\n")
    if(CRValue<=0.1):
        if(printComp): st.write("Matrix reasonably consistent, we could continue")
        return True
    else:
        if(printComp): st.write("Consistency Ratio is greater than 10%, we need to revise the subjective judgment")
        return False

#Parameter: matrix = Matrix yang akan dihitung konsistensinya, printComp = opsi untuk menampilkan komputasi konsistensi matrix
def pairwiseComp(matrix, printComp=True):
    matrix_len = len(matrix)
    
    #menghitung fuzzy geometric mean value
    geoMean = np.zeros((matrix_len,3))

    for i in range(matrix_len):
        for j in range(3):
            temp = 1
            for tfn in matrix[i]:
                temp *= tfn[j]
            temp = pow(temp, 1/matrix_len)
            geoMean[i,j] = temp

    if(printComp): 
        st.write("Fuzzy Geometric Mean Value: \n", geoMean, "\n")

    #menghitung total fuzzy geometric mean value
    geoMean_sum = np.sum(geoMean, axis=0)

    if(printComp): 
        st.write("Fuzzy Geometric Mean Sum:", geoMean_sum, "\n")

    #menghitung weights
    weights = np.zeros(matrix_len)

    for i in range(matrix_len):
        weights[i] = np.sum(geoMean[i] / geoMean_sum)

    if(printComp): 
        st.write("Weights: \n", weights, "\n")

    #menghitung normalized weights
    normWeights = weights / np.sum(weights)

    if(printComp): 
        st.write("Normalized Weights: ", normWeights,"\n")

    return normWeights

#Parameter: crxcr = Pairwise comparison matrix criteria X criteria, altxalt = Pairwise comparison matrices alternatif X alternatif , 
#       alternativesName = Nama dari setiap alternatif, printComp = opsi untuk menampilkan komputasi konsistensi matrix
#       show_criteria_matrix = opsi untuk menampilkan matrix pairwise kriteria, criteriaDict = nama kriteria
def FAHP(crxcr, altxalt, alternativesName, printComp=True, show_criteria_matrix=False, criteriaDict=None):
    

    # Cek konsistensi pairwise comparison matrix criteria x criteria
    if(printComp): st.write(f'<p style="font-size:28px">MENGHITUNG KONSISTENSI MATRIKS : \n</p>', unsafe_allow_html=True)
    crxcr_cons = isConsistent(crxcr, False)
    if(crxcr_cons):
        if(printComp): st.write("criteria X criteria comparison matrix reasonably consistent, we could continue")
    else: 
        if(printComp): st.write("criteria X criteria comparison matrix consistency ratio is greater than 10%, we need to revise the subjective judgment")

    # Cek konsistensi pairwise comparison matrix alternative x alternative untuk setiap criteria
    for i, altxalt_cr in enumerate(altxalt):
        isConsistent(altxalt_cr, False)
        if(crxcr_cons):
            if(printComp): st.write("alternatives X alternatives comparison matrix for criteria",i+1," is reasonably consistent, we could continue")
        else: 
            if(printComp): st.write("alternatives X alternatives comparison matrix for criteria",i+1,"'s consistency ratio is greater than 10%, we need to revise the subjective judgment")

    if(printComp): st.write("\n")

    if(printComp): st.write(f'<p style="font-size:28px">KRITERIA X KRITERIA : \n</p>', unsafe_allow_html=True)
    
    # Menampilkan matrix pairwise kriteria jika checkbox dicentang
    if show_criteria_matrix and criteriaDict is not None:
        display_criteria_pairwise_matrix(crxcr, criteriaDict)
    
    # Hitung nilai pairwise comparison weight untuk criteria x criteria
    crxcr_weights = pairwiseComp(crxcr, printComp)
    if(printComp): st.write("criteria X criteria weights: ", crxcr_weights)

    if(printComp): st.write("\n")
    if(printComp): st.write(f'<p style="font-size:28px">ALTERNATIF X ALTERNATIF : \n</p>', unsafe_allow_html=True)

    # Hitung nilai pairwise comparison weight untuk setiap alternative x alternative dalam setiap criteria
    altxalt_weights = np.zeros((len(altxalt),len(altxalt[0])))
    for i, altxalt_cr in enumerate(altxalt):
        if(printComp): st.write("alternative x alternative untuk criteria", criteriaDict[i],"\n")
        altxalt_weights[i] =  pairwiseComp(altxalt_cr, printComp)

    # Transpose matrix altxalt_weights
    altxalt_weights = altxalt_weights.transpose(1, 0)
    if(printComp): st.write("alternative x alternative weights:")
    if(printComp): st.write(altxalt_weights)

    # Hitung nilai jumlah dari perkalian crxcr_weights dengan altxalt_weights pada setiap kolom
    sumProduct = np.zeros(len(altxalt[0]))
    for i  in range(len(altxalt[0])):
        sumProduct[i] = np.dot(crxcr_weights, altxalt_weights[i])

    # Buat output dataframe
    output_df = pd.DataFrame(data=[alternativesName, sumProduct]).T
    output_df = output_df.rename(columns={0: "Alternatif", 1: "Score"})
    output_df = output_df.sort_values(by=['Score'],ascending = False)
    output_df.index = np.arange(1,len(output_df)+1)

    # Simpan DataFrame ke dalam file CSV
    output_df.to_csv("\n output_fahp.csv", index=False)

    return output_df

# Membuat fungsi untuk pengelompokan berdasarkan score dan batas skor yang ditentukan oleh pengguna
def kelompokkan_score(Score):
    if Score >= keringanan_50:
        return 'Keringanan 50%'
    elif Score >= keringanan_30:
        return 'Keringanan 30%'
    elif Score >= keringanan_20:
        return 'Keringanan 20%'
    else:
        return 'Tanpa Keringanan'

# Fungsi untuk menampilkan pairwise comparison matrix kriteria x kriteria
def display_criteria_pairwise_matrix(crxcr, criteriaDict):
    """
    Menampilkan pairwise comparison matrix kriteria x kriteria dalam bentuk tabel yang mudah dibaca
    
    Parameters:
    crxcr: numpy array - Matrix pairwise comparison kriteria x kriteria
    criteriaDict: array - Nama-nama kriteria
    """
    n_criteria = len(criteriaDict)
    
    # Membuat dataframe untuk lower bound (nilai minimum)
    lower_matrix = pd.DataFrame(
        data=[[crxcr[i][j][0] for j in range(n_criteria)] for i in range(n_criteria)],
        index=criteriaDict,
        columns=criteriaDict
    )
    
    # Membuat dataframe untuk middle value (nilai tengah)
    middle_matrix = pd.DataFrame(
        data=[[crxcr[i][j][1] for j in range(n_criteria)] for i in range(n_criteria)],
        index=criteriaDict,
        columns=criteriaDict
    )
    
    # Membuat dataframe untuk upper bound (nilai maksimum)
    upper_matrix = pd.DataFrame(
        data=[[crxcr[i][j][2] for j in range(n_criteria)] for i in range(n_criteria)],
        index=criteriaDict,
        columns=criteriaDict
    )
    
    # Membuat dataframe gabungan dengan format (l, m, u)
    combined_matrix = pd.DataFrame(
        index=criteriaDict,
        columns=criteriaDict
    )
    
    for i in range(n_criteria):
        for j in range(n_criteria):
            l = crxcr[i][j][0]
            m = crxcr[i][j][1] 
            u = crxcr[i][j][2]
            
            # Format angka dengan 3 desimal
            if l == m == u:
                combined_matrix.iloc[i, j] = f"{l:.0f}"
            else:
                combined_matrix.iloc[i, j] = f"({l:.3f}, {m:.3f}, {u:.3f})"
    
    st.subheader("📊 Pairwise Comparison Matrix Kriteria x Kriteria")
    st.write("Matrix ini menunjukkan perbandingan tingkat kepentingan antar kriteria menggunakan Triangular Fuzzy Numbers (TFN)")
    
    # Tampilkan matrix gabungan
    st.write("**Matrix Pairwise Comparison (Lower, Middle, Upper):**")
    st.dataframe(combined_matrix, use_container_width=True)
    
    # Tampilkan penjelasan
    with st.expander("ℹ️ Penjelasan Matrix"):
        st.write("""
        **Cara membaca matrix:**
        - Diagonal utama selalu bernilai (1, 1, 1) karena kriteria dibandingkan dengan dirinya sendiri
        - Nilai (l, m, u) menunjukkan tingkat kepentingan kriteria baris terhadap kriteria kolom
        - l = lower bound (batas bawah)
        - m = middle value (nilai tengah) 
        - u = upper bound (batas atas)
        - Semakin besar nilai, semakin penting kriteria baris dibanding kriteria kolom
        """)
        
        st.write("**Interpretasi nilai:**")
        st.write("- (1, 1, 1): Sama penting")
        st.write("- (1, 3, 5): Sedikit lebih penting")
        st.write("- (3, 5, 7): Lebih penting")
        st.write("- (5, 7, 9): Sangat penting")
        st.write("- (7, 9, 9): Mutlak lebih penting")

st.title("Fuzzy AHP untuk Seleksi Keringanan UKT")

with st.sidebar:
    st.write("## Upload File \n")
    st.write('Sampel file dapat diakses [disini!](https://github.com/tememumtaza/FuzzyAHP/tree/main/Data%20Sample)\n')
    file_criteria = st.file_uploader("Upload File Nilai Kriteria", type=['xlsx'], key="criteria")
    file_alternatives = st.file_uploader("Upload File Nilai Alternatif", type=['xlsx'], key="alternatives")

st.sidebar.markdown(" © 2023 Github [@temamumtaza](https://github.com/temamumtaza)")

if file_criteria is not None and file_alternatives is not None:
    criteriaDict = read_excel_file(file_criteria, 0)
    alternativesName = read_excel_file(file_alternatives, 0)

    criteria = read_excel_file(file_criteria, 1)
    for i in range(1, len(criteriaDict)+1):
        exec(f"altc{i} = read_excel_file(file_alternatives, {i})")

    def compare(*items):
        n = len(items)
        matrix = np.zeros((n, n, 3))
        for i, (c_i, v_i) in enumerate(items):
            for j, (c_j, v_j) in enumerate(items):
                if c_i == c_j or v_i == v_j:
                    matrix[i][j] = [1, 1, 3]
                else:
                    diff = abs(v_i - v_j)
                    if diff == 1:
                        matrix[i][j] = [1, 3, 5]
                    elif diff == 2:
                        matrix[i][j] = [3, 5, 7]
                    elif diff == 3:
                        matrix[i][j] = [5, 7, 9]
                    elif diff >= 4:
                        matrix[i][j] = [7, 9, 9]
                    if v_i < v_j:
                        matrix[i][j] = 1 / matrix[i][j][::-1]
        return matrix
    
    crxcr = np.array(compare(*criteria))
    for i in range(1, len(criteriaDict)+1):
        alt = eval(f"altc{i}")
        cr = compare(*alt)
        exec(f"altxalt_cr{i} = np.array(cr)")
    
    #Membuat array numpy untuk altxalt dengan mengambil nilai dari variabel global
    altxalt = np.stack([globals()[f"altxalt_cr{i+1}"] for i in range(len(criteriaDict))])

    # Membuat checkbox untuk menampilkan perhitungan lengkap (konsistensi matrix)
    show_comp = st.checkbox("Tampilkan Penghitungan Fuzzy AHP")
    
    # Membuat checkbox untuk menampilkan matrix pairwise kriteria
    show_criteria_matrix = st.checkbox("Tampilkan Matrix Pairwise Kriteria")

    #Memanggil fungsi FAHP dengan parameter yang telah didefinisikan sebelumnya
    #printComp di-set False agar tidak menampilkan komputasi konsistensi matrix
    output = FAHP(crxcr, altxalt, alternativesName, show_comp, show_criteria_matrix, criteriaDict)
    
    #Menampilkan rangking alternatif dengan output dari fungsi FAHP
    st.write("\n RANGKING ALTERNATIF:\n", output)

    # Tampilkan widget untuk memilih opsi pengelompokan
    pengelompokan_option = st.radio("Pilih opsi pengelompokan:", ("Alokasi Persentase", "Batas Skor"))

    # Jika opsi yang dipilih adalah Alokasi Persentase
    if pengelompokan_option == "Alokasi Persentase":
        st.header("Pengelompokan Data Berdasarkan Alokasi Persentase")
        
        # Tambahkan widget untuk memungkinkan pengguna mengatur kuota pengaju keringanan
        kuota_pengaju = st.slider('Kuota Pengaju Keringanan:', min_value=0, max_value=len(output), value=180, step=1)

        # Tambahkan widget untuk memungkinkan pengguna mengatur alokasi persentase untuk masing-masing kelompok
        keringanan_50 = st.slider('Alokasi Keringanan 50%:', min_value=0, max_value=100, value=20, step=1)
        keringanan_30 = st.slider('Alokasi Keringanan 30%:', min_value=0, max_value=100, value=30, step=1)
        keringanan_20 = st.slider('Alokasi Keringanan 20%:', min_value=0, max_value=100, value=50, step=1)

        # Urutkan data berdasarkan skor secara descending
        output = output.sort_values(by='Score', ascending=False)

        # Potong data sesuai dengan kuota yang ditentukan oleh pengguna
        output_kuota = output.iloc[:kuota_pengaju]
        output_tidak_kuota = output.iloc[kuota_pengaju:]

        # Menghitung jumlah data untuk masing-masing kelompok
        total = len(output_kuota)

        # Menghitung kuota untuk masing-masing kelompok
        kuota_50 = int(total * keringanan_50 / 100)
        kuota_30 = int(total * keringanan_30 / 100)
        kuota_20 = int(total * keringanan_20 / 100)

        # Melakukan pengelompokan berdasarkan kuota
        output_kuota.loc[output_kuota.index[:kuota_50], 'kelompok'] = 'Keringanan 50%'
        output_kuota.loc[output_kuota.index[kuota_50:kuota_50 + kuota_30], 'kelompok'] = 'Keringanan 30%'
        output_kuota.loc[output_kuota.index[kuota_50 + kuota_30:], 'kelompok'] = 'Keringanan 20%'

        # Tambahkan status "Tidak dapat keringanan" untuk data yang tidak memenuhi kuota
        output_tidak_kuota['kelompok'] = 'Tidak dapat keringanan'

        # Gabungkan output_kuota dan output_tidak_kuota
        output_final = pd.concat([output_kuota, output_tidak_kuota])

        # Menampilkan dataframe yang sudah diurutkan dan dikelompokkan
        st.write(output_final)

        # Menghitung jumlah mahasiswa pada setiap kelompok
        jumlah_keringanan_50 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 50%'])
        jumlah_keringanan_30 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 30%'])
        jumlah_keringanan_20 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 20%'])
        jumlah_tidak_keringanan = len(output_tidak_kuota)

        # Membuat diagram pie
        fig = go.Figure(data=[go.Pie(labels=['Keringanan 50%', 'Keringanan 30%', 'Keringanan 20%', 'Tidak dapat keringanan'], values=[jumlah_keringanan_50, jumlah_keringanan_30, jumlah_keringanan_20, jumlah_tidak_keringanan])])
        fig.update_layout(title='Jumlah Mahasiswa pada Tiap Kelompok Keringanan')
        st.plotly_chart(fig, use_container_width=True)
    
    # Jika opsi yang dipilih adalah Batas Skor
    else:
        st.header("Pengelompokan Data Berdasarkan Skor Tertinggi")
    
        # Menambahkan widget untuk memungkinkan pengguna menentukan batas skor untuk masing-masing kelompok
        keringanan_50 = st.slider('Batas skor Keringanan 50%:', min_value=0.00, max_value=0.01, value=0.0056,step=0.0001, format="%.4f")
        keringanan_30 = st.slider('Batas skor Keringanan 30%:', min_value=0.00, max_value=0.01, value=0.0048,step=0.0001, format="%.4f")
        keringanan_20 = st.slider('Batas skor Keringanan 20%:', min_value=0.00, max_value=0.01, value=0.0035,step=0.0001, format="%.4f")
        
        # Melakukan pengelompokan dan pengurutan dataframe
        output['kelompok'] = output['Score'].apply(kelompokkan_score)
        output = output.sort_values(by='Score', ascending=False)

        # Menghitung presentase untuk masing-masing kelompok
        count = output.groupby('kelompok')[output.columns[0]].count()
        labels = count.index.tolist()
        values = count.values.tolist()
        total = sum(values)
        percentages = [round(value/total*100,2) for value in values]

        # Menampilkan dataframe yang sudah diurutkan dan dikelompokkan
        st.write(output)

        # Membuat diagram pie
        fig = go.Figure(data=[go.Pie(labels=labels, values=percentages)])
        fig.update_layout(title='Presentase Kelompok Keringanan')
        st.plotly_chart(fig, use_container_width=True)

    # st.markdown(filedownload(output_final), unsafe_allow_html=True)

else:
    st.write("Mohon upload kedua file terlebih dahulu.")