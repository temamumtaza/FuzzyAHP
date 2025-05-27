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

def fuzzy_consistency_check(matrix, printComp=True):
    """
    Pengecekan konsistensi untuk Triangular Fuzzy Numbers (TFN) menggunakan metode Alpha-Cut
    
    Parameters:
    matrix: array of TFN - Matrix perbandingan berpasangan dalam format TFN [(l,m,u)]
    printComp: bool - Opsi untuk menampilkan detail perhitungan
    
    Returns:
    dict: Contains consistency results using alpha-cut method (alpha=1)
    """
    mat_len = len(matrix)
    RI = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    # Metode Alpha-Cut dengan alpha = 1: ambil nilai tengah (m) dari TFN (l, m, u)
    crisp_matrix = np.array([[tfn[1] for tfn in row] for row in matrix])
    
    if printComp:
        st.markdown("#### 🔢 **Transformasi TFN ke Crisp Matrix (Alpha-Cut α=1):**")
        st.write("**Matrix TFN Original:**")
        
        # Tampilkan matrix TFN dalam format yang mudah dibaca
        tfn_display = []
        for row in matrix:
            tfn_row = []
            for tfn in row:
                l, m, u = tfn
                if l == m == u:
                    tfn_row.append(f"{m:.0f}")
                else:
                    tfn_row.append(f"({l:.1f},{m:.1f},{u:.1f})")
            tfn_display.append(tfn_row)
        
        st.dataframe(pd.DataFrame(tfn_display), use_container_width=True)
        
        st.write("**Crisp Matrix (nilai tengah/median dari TFN):**")
        st.write(pd.DataFrame(crisp_matrix).round(3))
        
        st.info("💡 **Alpha-Cut α=1:** Mengambil nilai tengah (m) dari setiap TFN (l,m,u) untuk transformasi ke crisp matrix")
    
    # Hitung konsistensi menggunakan crisp matrix
    def calculate_consistency_alpha_cut():
        # Validasi matrix - pastikan tidak ada nilai negatif atau nol pada diagonal
        if np.any(np.diag(crisp_matrix) <= 0):
            if printComp:
                st.warning("⚠️ Matrix memiliki diagonal yang tidak valid")
            return {
                'lambda_max': mat_len,
                'CI': 0,
                'RI': 0,
                'CR': 0,
                'consistent': True
            }
        
        # Perhitungan eigenvalue untuk crisp matrix
        try:
            eigenvalues = np.linalg.eigvals(crisp_matrix)
            # Filter hanya eigenvalue real dan positif
            real_eigenvalues = [ev.real for ev in eigenvalues if ev.imag == 0 and ev.real > 0]
            
            if len(real_eigenvalues) == 0:
                lambda_max = mat_len  # Default ke n jika tidak ada eigenvalue valid
            else:
                lambda_max = max(real_eigenvalues)
                
            # Pastikan lambda_max >= n (properti fundamental dari matriks pairwise comparison)
            lambda_max = max(lambda_max, mat_len)
            
        except np.linalg.LinAlgError:
            if printComp:
                st.warning("⚠️ Error perhitungan eigenvalue untuk crisp matrix")
            lambda_max = mat_len
        
        if mat_len >= 10:
            ri_value = RI[10]
        else:
            ri_value = RI[mat_len] if mat_len in RI else 0
        
        ci_value = (lambda_max - mat_len) / (mat_len - 1) if mat_len > 1 else 0
        cr_value = ci_value / ri_value if ri_value > 0 else 0
        
        return {
            'lambda_max': lambda_max,
            'CI': ci_value,
            'RI': ri_value,
            'CR': cr_value,
            'consistent': cr_value <= 0.1
        }
    
    # Hitung konsistensi menggunakan alpha-cut
    alpha_cut_result = calculate_consistency_alpha_cut()
    
    if printComp:
        st.markdown("#### 📊 **Hasil Analisis Konsistensi (Alpha-Cut Method):**")
        
        # Tabel hasil konsistensi
        consistency_df = pd.DataFrame({
            'Metode': ['Alpha-Cut (α=1)'],
            'λ_max': [alpha_cut_result['lambda_max']],
            'CI': [alpha_cut_result['CI']],
            'RI': [alpha_cut_result['RI']],
            'CR': [alpha_cut_result['CR']],
            'Status': ['✅ Konsisten' if alpha_cut_result['consistent'] else '❌ Tidak Konsisten']
        })
        
        st.dataframe(consistency_df.round(4), use_container_width=True)
        
        # Overall assessment
        st.markdown("#### 🏆 **Penilaian Konsistensi:**")
        
        if alpha_cut_result['consistent']:
            st.success("✅ **Matrix TFN KONSISTEN** - Dapat digunakan untuk perhitungan lanjutan")
        else:
            st.warning("⚠️ **Matrix TFN TIDAK KONSISTEN** - Disarankan untuk merevisi penilaian")
        
        # Detailed explanation
        with st.expander("🔍 **Penjelasan Metode Alpha-Cut untuk Konsistensi TFN**"):
            st.write("""
            **Metode Alpha-Cut (α=1):**
            
            1. **Transformasi TFN ke Crisp:** Menggunakan alpha-cut dengan α=1
               - TFN (l, m, u) → m (nilai tengah/median)
               - Memberikan representasi crisp yang deterministik dari TFN
            
            2. **Perhitungan Konsistensi:** Menggunakan metode AHP klasik pada crisp matrix
               - Consistency Index (CI) = (λmax - n) / (n - 1)
               - Consistency Ratio (CR) = CI / Random Index (RI)
               - Kriteria: CR ≤ 0.1 = Konsisten
            
            **Kelebihan Metode Alpha-Cut:**
            - Sederhana dan mudah dipahami
            - Komputasi efisien 
            - Konsisten dengan teori AHP klasik
            - Memberikan hasil yang deterministik
            
            **Logic TFN yang Digunakan:**
            - **(1,1,3):** Sama penting atau elemen diagonal
            - **(1,3,5):** Sedikit lebih penting 
            - **(3,5,7):** Lebih penting 
            - **(5,7,9):** Sangat penting 
            - **(7,9,9):** Mutlak lebih penting
            """)
    
    return {
        'alpha_cut': alpha_cut_result,
        'overall_consistent': alpha_cut_result['consistent'],
        'crisp_matrix': crisp_matrix
    }

def isConsistent(matrix, printComp=True):
    """
    Legacy function for backward compatibility - now uses fuzzy consistency check
    """
    # Gunakan printComp=False untuk menghindari duplikasi tampilan
    # Karena pesan konsistensi akan ditampilkan di fungsi FAHP()
    result = fuzzy_consistency_check(matrix, printComp=False)
    return result['overall_consistent']

#Parameter: matrix = Matrix yang akan dihitung konsistensinya, printComp = opsi untuk menampilkan komputasi konsistensi matrix
#          labels = Array nama untuk kolom (kriteria atau alternatif)
def pairwiseComp(matrix, printComp=True, labels=None):
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
        # Tampilkan Fuzzy Geometric Mean Value dengan nama kolom yang deskriptif
        geo_mean_df = pd.DataFrame(geoMean, columns=['l', 'm', 'u'])
        st.write("**Fuzzy Geometric Mean Value:**")
        st.dataframe(geo_mean_df.round(4), use_container_width=True)

    #menghitung total fuzzy geometric mean value
    geoMean_sum = np.sum(geoMean, axis=0)

    if(printComp): 
        # Tampilkan Fuzzy Geometric Mean Sum dengan format yang lebih rapi
        geo_mean_sum_df = pd.DataFrame([geoMean_sum], columns=['l', 'm', 'u'])
        st.write("**Fuzzy Geometric Mean Sum:**")
        st.dataframe(geo_mean_sum_df.round(4), use_container_width=True)

    #menghitung weights
    weights = np.zeros(matrix_len)

    for i in range(matrix_len):
        weights[i] = np.sum(geoMean[i] / geoMean_sum)

    if(printComp): 
        # Tampilkan Weights dengan format yang lebih rapi
        if labels is not None:
            column_labels = labels
        else:
            column_labels = [f'Item {i+1}' for i in range(len(weights))]
        
        weights_df = pd.DataFrame([weights], columns=column_labels)
        st.write("**Weights:**")
        st.dataframe(weights_df.round(4), use_container_width=True)

    #menghitung normalized weights
    normWeights = weights / np.sum(weights)

    if(printComp): 
        # Tampilkan Normalized Weights dengan format yang lebih rapi
        if labels is not None:
            column_labels = labels
        else:
            column_labels = [f'Item {i+1}' for i in range(len(normWeights))]
            
        norm_weights_df = pd.DataFrame([normWeights], columns=column_labels)
        st.write("**Normalized Weights:**")
        st.dataframe(norm_weights_df.round(4), use_container_width=True)

    return normWeights

#Parameter: crxcr = Pairwise comparison matrix criteria X criteria, altxalt = Pairwise comparison matrices alternatif X alternatif , 
#       alternativesName = Nama dari setiap alternatif, printComp = opsi untuk menampilkan komputasi konsistensi matrix
#       show_criteria_matrix = opsi untuk menampilkan matrix pairwise kriteria, criteriaDict = nama kriteria
def FAHP(crxcr, altxalt, alternativesName, printComp=True, show_criteria_matrix=False, criteriaDict=None):
    

    # Cek konsistensi pairwise comparison matrix criteria x criteria
    if(printComp): 
        st.markdown("---")
        st.markdown("## 🔍 **TAHAP 1: PENGECEKAN KONSISTENSI MATRIKS**")
        
        with st.expander("ℹ️ Penjelasan Pengecekan Konsistensi TFN", expanded=False):
            st.write("""
            **Tujuan:** Memastikan bahwa matriks perbandingan berpasangan TFN konsisten dan dapat diandalkan.
            
            **Metode Alpha-Cut (α=1) yang Digunakan:**
            1. **Transformasi TFN ke Crisp:** Menggunakan alpha-cut dengan α=1
               - TFN (l, m, u) → m (nilai tengah/median)
               - Memberikan representasi crisp yang deterministik dari TFN
            
            2. **Perhitungan Konsistensi:** Menggunakan metode AHP klasik pada crisp matrix
               - Consistency Index (CI) = (λmax - n) / (n - 1)
               - Consistency Ratio (CR) = CI / Random Index (RI)
               - Kriteria: CR ≤ 0.1 = Konsisten
            
            **Keunggulan Metode Alpha-Cut:**
            - Sederhana dan mudah dipahami
            - Komputasi efisien 
            - Konsisten dengan teori AHP klasik
            - Memberikan hasil yang deterministik
            
            """)
        
        st.markdown("### 📊 **Konsistensi Matrix Kriteria x Kriteria:**")
    
    crxcr_cons = isConsistent(crxcr, printComp)
    if(crxcr_cons):
        if(printComp): 
            st.success("✅ Matrix kriteria x kriteria konsisten (CR ≤ 0.1). Perhitungan dapat dilanjutkan.")
    else: 
        if(printComp): 
            st.warning("⚠️ Matrix kriteria x kriteria tidak konsisten (CR > 0.1). Namun perhitungan tetap dilanjutkan.")

    # Cek konsistensi pairwise comparison matrix alternative x alternative untuk setiap criteria
    if(printComp): st.markdown("### 📊 **Konsistensi Matrix Alternatif x Alternatif:**")
    
    for i, altxalt_cr in enumerate(altxalt):
        altxalt_cons = isConsistent(altxalt_cr, False)
        if(printComp):
            if(altxalt_cons):
                st.success(f"✅ Matrix alternatif untuk kriteria '{criteriaDict[i]}' konsisten")
            else: 
                st.warning(f"⚠️ Matrix alternatif untuk kriteria '{criteriaDict[i]}' tidak konsisten")

    if(printComp): 
        st.markdown("---")
        st.markdown("## ⚖️ **TAHAP 2: PERHITUNGAN BOBOT KRITERIA**")
        
        with st.expander("ℹ️ Penjelasan Perhitungan Bobot Kriteria", expanded=False):
            st.write("""
            **Tujuan:** Menghitung bobot (prioritas) setiap kriteria berdasarkan matriks perbandingan berpasangan.
            
            **Metode Fuzzy Geometric Mean:**
            1. **Fuzzy Geometric Mean** = ∜(a₁₁ × a₁₂ × ... × a₁ₙ) untuk setiap baris
            2. **Fuzzy Weight** = Fuzzy Geometric Mean / Σ(Fuzzy Geometric Mean)
            3. **Normalisasi** untuk mendapatkan bobot akhir
            
            **Hasil:** Bobot setiap kriteria yang akan digunakan untuk menghitung skor akhir alternatif.
            """)
    
    # Menampilkan matrix pairwise kriteria jika checkbox dicentang
    if show_criteria_matrix and criteriaDict is not None:
        display_criteria_pairwise_matrix(crxcr, criteriaDict)
    
    # Hitung nilai pairwise comparison weight untuk criteria x criteria
    crxcr_weights = pairwiseComp(crxcr, printComp, criteriaDict)
    
    if(printComp): 
        st.markdown("### 🎯 **Bobot Final Kriteria:**")
        
        # Membuat DataFrame untuk menampilkan bobot kriteria dengan lebih rapi
        weights_df = pd.DataFrame({
            'Kriteria': criteriaDict,
            'Bobot': crxcr_weights,
            'Persentase': [f"{w*100:.2f}%" for w in crxcr_weights]
        })
        weights_df.index = weights_df.index + 1
        
        st.dataframe(weights_df, use_container_width=True)
        
        # Tambahkan visualisasi bobot kriteria
        import plotly.express as px
        fig_weights = px.pie(
            values=crxcr_weights, 
            names=criteriaDict,
            title="📊 Distribusi Bobot Kriteria"
        )
        fig_weights.update_traces(textinfo='label+percent')
        st.plotly_chart(fig_weights, use_container_width=True)

    if(printComp): 
        st.markdown("---")
        st.markdown("## 🏆 **TAHAP 3: PERHITUNGAN BOBOT ALTERNATIF**")
        
        with st.expander("ℹ️ Penjelasan Perhitungan Bobot Alternatif", expanded=False):
            st.write("""
            **Tujuan:** Menghitung bobot setiap alternatif terhadap masing-masing kriteria.
            
            **Proses:**
            1. Untuk setiap kriteria, buat matriks perbandingan berpasangan alternatif
            2. Hitung bobot alternatif menggunakan Fuzzy Geometric Mean
            3. Normalisasi bobot untuk setiap kriteria
            
            **Hasil:** Matriks bobot alternatif dimana setiap kolom mewakili bobot alternatif untuk kriteria tertentu.
            """)

    # Hitung nilai pairwise comparison weight untuk setiap alternative x alternative dalam setiap criteria
    altxalt_weights = np.zeros((len(altxalt),len(altxalt[0])))
    for i, altxalt_cr in enumerate(altxalt):
        if(printComp): 
            st.markdown(f"### 📋 **Bobot Alternatif untuk Kriteria: {criteriaDict[i]}**")
        altxalt_weights[i] =  pairwiseComp(altxalt_cr, printComp, alternativesName)

    # Transpose matrix altxalt_weights
    altxalt_weights = altxalt_weights.transpose(1, 0)
    
    if(printComp): 
        st.markdown("### 📊 **Ringkasan Bobot Alternatif untuk Semua Kriteria:**")
        
        # Membuat DataFrame untuk menampilkan bobot alternatif dengan lebih rapi
        altxalt_df = pd.DataFrame(
            altxalt_weights,
            index=alternativesName,
            columns=criteriaDict
        )
        
        st.dataframe(altxalt_df.round(4), use_container_width=True)
        
        with st.expander("💡 Interpretasi Tabel Bobot Alternatif"):
            st.write("""
            - **Baris**: Menunjukkan alternatif (mahasiswa)
            - **Kolom**: Menunjukkan kriteria
            - **Nilai**: Bobot alternatif terhadap kriteria tertentu (0-1)
            - **Semakin tinggi nilai**: Semakin baik alternatif tersebut pada kriteria yang bersangkutan
            """)

    # Hitung nilai jumlah dari perkalian crxcr_weights dengan altxalt_weights pada setiap kolom
    if(printComp): 
        st.markdown("---")
        st.markdown("## 🎯 **TAHAP 4: PERHITUNGAN SKOR AKHIR**")
        
        with st.expander("ℹ️ Penjelasan Perhitungan Skor Akhir", expanded=False):
            st.write("""
            **Tujuan:** Menghitung skor akhir setiap alternatif dengan mempertimbangkan bobot kriteria.
            
            **Formula:** 
            Skor Akhir = Σ(Bobot Kriteria × Bobot Alternatif)
            
            **Proses:**
            1. Kalikan bobot setiap kriteria dengan bobot alternatif pada kriteria tersebut
            2. Jumlahkan semua hasil perkalian
            3. Urutkan alternatif berdasarkan skor tertinggi
            
            **Hasil:** Ranking alternatif berdasarkan skor Fuzzy AHP
            """)
    
    sumProduct = np.zeros(len(altxalt[0]))
    for i  in range(len(altxalt[0])):
        sumProduct[i] = np.dot(crxcr_weights, altxalt_weights[i])

    # Buat output dataframe
    output_df = pd.DataFrame(data=[alternativesName, sumProduct]).T
    output_df = output_df.rename(columns={0: "Alternatif", 1: "Score"})
    output_df = output_df.sort_values(by=['Score'],ascending = False)
    output_df.index = np.arange(1,len(output_df)+1)
    
    if(printComp):
        st.markdown("### 🏆 **Detail Perhitungan Skor:**")
        
        # Membuat tabel detail perhitungan untuk beberapa alternatif teratas
        detail_df = pd.DataFrame(index=alternativesName[:10])  # Tampilkan 10 teratas saja
        
        for j, criteria in enumerate(criteriaDict):
            detail_df[f'{criteria}\n(Bobot: {crxcr_weights[j]:.3f})'] = [
                f"{altxalt_weights[i][j]:.3f} × {crxcr_weights[j]:.3f} = {altxalt_weights[i][j] * crxcr_weights[j]:.4f}"
                for i in range(min(10, len(alternativesName)))
            ]
        
        detail_df['Total Score'] = [f"{sumProduct[i]:.4f}" for i in range(min(10, len(alternativesName)))]
        
        st.dataframe(detail_df, use_container_width=True)
        
        st.info("💡 **Tabel di atas menunjukkan:** Bobot Alternatif × Bobot Kriteria = Kontribusi ke Skor Total (untuk 10 alternatif teratas)")

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
        - Diagonal utama dan elemen yang sama bernilai (1, 1, 3)
        - Nilai (l, m, u) menunjukkan tingkat kepentingan kriteria baris terhadap kriteria kolom
        - l = lower bound (batas bawah)
        - m = middle value (nilai tengah) - digunakan untuk alpha-cut
        - u = upper bound (batas atas)
        - Semakin besar nilai, semakin penting kriteria baris dibanding kriteria kolom
        """)
        
        st.write("**Logic TFN yang Digunakan:**")
        st.write("- **(1, 1, 3):** Elemen diagonal atau sama penting")
        st.write("- **(1, 3, 5):** Sedikit lebih penting (selisih nilai = 1)")
        st.write("- **(3, 5, 7):** Lebih penting (selisih nilai = 2)")
        st.write("- **(5, 7, 9):** Sangat penting (selisih nilai = 3)")
        st.write("- **(7, 9, 9):** Mutlak lebih penting (selisih nilai ≥ 4)")

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
                if i == j or c_i == c_j or v_i == v_j:
                    # Diagonal elements atau elemen yang sama
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
                    
                    # Jika v_i < v_j, maka kita perlu invers TFN
                    if v_i < v_j:
                        l, m, u = matrix[i][j]
                        # Invers TFN: [1/u, 1/m, 1/l]
                        matrix[i][j] = [1/u, 1/m, 1/l]
        return matrix
    
    crxcr = np.array(compare(*criteria))
    for i in range(1, len(criteriaDict)+1):
        alt = eval(f"altc{i}")
        cr = compare(*alt)
        exec(f"altxalt_cr{i} = np.array(cr)")
    
    #Membuat array numpy untuk altxalt dengan mengambil nilai dari variabel global
    altxalt = np.stack([globals()[f"altxalt_cr{i+1}"] for i in range(len(criteriaDict))])

    # Membuat checkbox untuk menampilkan perhitungan lengkap (termasuk matrix pairwise dan konsistensi)
    show_comp = st.checkbox("🔍 Tampilkan Detail Perhitungan Fuzzy AHP (termasuk Matrix Pairwise)")

    #Memanggil fungsi FAHP dengan parameter yang telah didefinisikan sebelumnya
    output = FAHP(crxcr, altxalt, alternativesName, show_comp, show_comp, criteriaDict)
    
    # Menampilkan rangking alternatif dengan output dari fungsi FAHP
    st.markdown("---")
    st.markdown("## 🏆 **HASIL RANKING ALTERNATIF**")
    
    with st.expander("ℹ️ Penjelasan Hasil Ranking", expanded=False):
        st.write("""
        **Hasil Akhir Fuzzy AHP:**
        - Tabel ini menunjukkan ranking alternatif berdasarkan skor Fuzzy AHP
        - Skor tertinggi menunjukkan alternatif terbaik berdasarkan kriteria yang ditetapkan
        - Ranking sudah mempertimbangkan bobot kepentingan setiap kriteria
        
        **Cara Membaca:**
        - **No.**: Ranking/peringkat alternatif
        - **Alternatif**: Nama alternatif (mahasiswa)
        - **Score**: Skor akhir hasil perhitungan Fuzzy AHP (0-1)
        """)
    
    # Memformat output dengan lebih rapi
    output_display = output.copy()
    output_display['Score'] = output_display['Score'].round(6)
    output_display.index.name = 'Ranking'
    
    st.dataframe(output_display, use_container_width=True)

    # Tampilkan widget untuk memilih opsi pengelompokan
    st.markdown("---")
    st.markdown("## 📊 **PENGELOMPOKAN KERINGANAN UKT**")
    
    with st.expander("ℹ️ Penjelasan Sistem Pengelompokan", expanded=False):
        st.write("""
        **Dua Metode Pengelompokan Tersedia:**
        
        **1. Alokasi Persentase:**
        - Berdasarkan kuota dan persentase yang ditetapkan
        - Misalnya: 20% dari 180 mahasiswa terbaik mendapat keringanan 50%
        - Cocok untuk budget/kuota yang sudah ditentukan
        
        **2. Batas Skor:**
        - Berdasarkan threshold skor minimum untuk setiap kategori
        - Misalnya: Skor ≥ 0.0056 mendapat keringanan 50%
        - Cocok untuk standar kualitas yang sudah ditetapkan
        """)
    
    pengelompokan_option = st.radio("🎯 **Pilih Metode Pengelompokan:**", ("Alokasi Persentase", "Batas Skor"))

    # Jika opsi yang dipilih adalah Alokasi Persentase
    if pengelompokan_option == "Alokasi Persentase":
        st.markdown("### 📊 **Metode: Alokasi Persentase**")
        
        with st.expander("💡 Cara Kerja Alokasi Persentase", expanded=False):
            st.write("""
            **Proses:**
            1. Tentukan total kuota mahasiswa yang berhak mendapat keringanan
            2. Tentukan persentase untuk setiap kategori keringanan
            3. Sistem akan mengambil mahasiswa dengan ranking terbaik sesuai kuota
            4. Pembagian kategori berdasarkan urutan ranking dan persentase yang ditetapkan
            """)
        
        # Tambahkan widget untuk memungkinkan pengguna mengatur kuota pengaju keringanan
        st.markdown("#### ⚙️ **Pengaturan Kuota dan Alokasi**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kuota_pengaju = st.slider(
                '👥 **Kuota Total Pengaju Keringanan:**', 
                min_value=0, 
                max_value=len(output), 
                value=min(180, len(output)), 
                step=1,
                help="Total mahasiswa yang berhak mendapat keringanan UKT"
            )

        with col2:
            st.metric("📈 **Total Mahasiswa**", len(output))
            if kuota_pengaju > 0:
                st.metric("📊 **Persentase dari Total**", f"{kuota_pengaju/len(output)*100:.1f}%")

        # Tambahkan widget untuk memungkinkan pengguna mengatur alokasi persentase untuk masing-masing kelompok
        st.markdown("#### 🎯 **Alokasi Persentase Keringanan**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            keringanan_50 = st.slider(
                '🥇 **Keringanan 50%:**', 
                min_value=0, 
                max_value=100, 
                value=20, 
                step=1,
                help="Persentase mahasiswa terbaik yang mendapat keringanan 50%"
            )
        
        with col2:
            keringanan_30 = st.slider(
                '🥈 **Keringanan 30%:**', 
                min_value=0, 
                max_value=100, 
                value=30, 
                step=1,
                help="Persentase mahasiswa yang mendapat keringanan 30%"
            )
        
        with col3:
            keringanan_20 = st.slider(
                '🥉 **Keringanan 20%:**', 
                min_value=0, 
                max_value=100, 
                value=50, 
                step=1,
                help="Persentase mahasiswa yang mendapat keringanan 20%"
            )
        
        # Validasi total persentase
        total_persen = keringanan_50 + keringanan_30 + keringanan_20
        if total_persen != 100:
            st.warning(f"⚠️ **Perhatian:** Total persentase = {total_persen}%. Silakan sesuaikan agar total = 100%")
        else:
            st.success("✅ **Total persentase = 100%** - Konfigurasi valid!")

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

        # Menampilkan ringkasan statistik
        st.markdown("#### 📈 **Ringkasan Hasil Pengelompokan**")
        
        # Menghitung jumlah mahasiswa pada setiap kelompok
        jumlah_keringanan_50 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 50%'])
        jumlah_keringanan_30 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 30%'])
        jumlah_keringanan_20 = len(output_kuota[output_kuota['kelompok'] == 'Keringanan 20%'])
        jumlah_tidak_keringanan = len(output_tidak_kuota)
        
        # Tampilkan metrics dalam columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🥇 **Keringanan 50%**", 
                jumlah_keringanan_50,
                delta=f"{jumlah_keringanan_50/len(output)*100:.1f}% dari total"
            )
        
        with col2:
            st.metric(
                "🥈 **Keringanan 30%**", 
                jumlah_keringanan_30,
                delta=f"{jumlah_keringanan_30/len(output)*100:.1f}% dari total"
            )
        
        with col3:
            st.metric(
                "🥉 **Keringanan 20%**", 
                jumlah_keringanan_20,
                delta=f"{jumlah_keringanan_20/len(output)*100:.1f}% dari total"
            )
        
        with col4:
            st.metric(
                "❌ **Tanpa Keringanan**", 
                jumlah_tidak_keringanan,
                delta=f"{jumlah_tidak_keringanan/len(output)*100:.1f}% dari total"
            )

        # Menampilkan dataframe yang sudah diurutkan dan dikelompokkan
        st.markdown("#### 📋 **Detail Hasil Pengelompokan**")
        
        # Format output_final dengan warna untuk setiap kategori
        output_final_display = output_final.copy()
        output_final_display['Score'] = output_final_display['Score'].round(6)
        output_final_display.index.name = 'Ranking'
        
        st.dataframe(output_final_display, use_container_width=True)

        # Membuat diagram pie yang lebih menarik
        st.markdown("#### 🥧 **Visualisasi Distribusi Keringanan**")
        
        fig = go.Figure(data=[go.Pie(
            labels=['Keringanan 50%', 'Keringanan 30%', 'Keringanan 20%', 'Tidak dapat keringanan'], 
            values=[jumlah_keringanan_50, jumlah_keringanan_30, jumlah_keringanan_20, jumlah_tidak_keringanan],
            hole=0.3,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )])
        fig.update_layout(
            title='📊 Distribusi Mahasiswa per Kategori Keringanan UKT',
            annotations=[dict(text='Total<br>' + str(len(output)), x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        fig.update_traces(textinfo='label+percent+value')
        st.plotly_chart(fig, use_container_width=True)
    
    # Jika opsi yang dipilih adalah Batas Skor
    else:
        st.markdown("### 📊 **Metode: Batas Skor**")
        
        with st.expander("💡 Cara Kerja Batas Skor", expanded=False):
            st.write("""
            **Proses:**
            1. Tentukan threshold (batas minimum) skor untuk setiap kategori keringanan
            2. Sistem akan mengelompokkan mahasiswa berdasarkan skor yang dicapai
            3. Mahasiswa dengan skor ≥ threshold akan masuk kategori keringanan yang sesuai
            4. Jumlah mahasiswa per kategori akan bervariasi tergantung distribusi skor
            """)
        
        # Dapatkan range skor untuk memberikan panduan
        min_score = output['Score'].min()
        max_score = output['Score'].max()
        
        st.info(f"📊 **Range Skor:** {min_score:.6f} - {max_score:.6f}")
    
        # Menambahkan widget untuk memungkinkan pengguna menentukan batas skor untuk masing-masing kelompok
        st.markdown("#### ⚙️ **Pengaturan Threshold Skor**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            keringanan_50 = st.slider(
                '🥇 **Threshold Keringanan 50%:**', 
                min_value=float(min_score), 
                max_value=float(max_score), 
                value=min(0.0056, float(max_score)),
                step=0.0001, 
                format="%.4f",
                help="Skor minimum untuk mendapat keringanan 50%"
            )
        
        with col2:
            keringanan_30 = st.slider(
                '🥈 **Threshold Keringanan 30%:**', 
                min_value=float(min_score), 
                max_value=float(keringanan_50), 
                value=min(0.0048, float(keringanan_50)),
                step=0.0001, 
                format="%.4f",
                help="Skor minimum untuk mendapat keringanan 30%"
            )
        
        with col3:
            keringanan_20 = st.slider(
                '🥉 **Threshold Keringanan 20%:**', 
                min_value=float(min_score), 
                max_value=float(keringanan_30), 
                value=min(0.0035, float(keringanan_30)),
                step=0.0001, 
                format="%.4f",
                help="Skor minimum untuk mendapat keringanan 20%"
            )
        
        # Melakukan pengelompokan dan pengurutan dataframe
        output['kelompok'] = output['Score'].apply(kelompokkan_score)
        output = output.sort_values(by='Score', ascending=False)

        # Menghitung jumlah dan persentase untuk masing-masing kelompok
        count = output.groupby('kelompok')[output.columns[0]].count()
        labels = count.index.tolist()
        values = count.values.tolist()
        total = sum(values)
        percentages = [round(value/total*100,2) for value in values]

        # Menampilkan ringkasan statistik
        st.markdown("#### 📈 **Ringkasan Hasil Pengelompokan**")
        
        # Buat mapping untuk warna dan ikon
        category_info = {
            'Keringanan 50%': {'icon': '🥇', 'color': '#1f77b4'},
            'Keringanan 30%': {'icon': '🥈', 'color': '#ff7f0e'},
            'Keringanan 20%': {'icon': '🥉', 'color': '#2ca02c'},
            'Tanpa Keringanan': {'icon': '❌', 'color': '#d62728'}
        }
        
        # Tampilkan metrics
        cols = st.columns(len(labels))
        for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
            with cols[i]:
                icon = category_info.get(label, {}).get('icon', '📊')
                st.metric(
                    f"{icon} **{label}**",
                    value,
                    delta=f"{percentage}% dari total"
                )

        # Menampilkan dataframe yang sudah diurutkan dan dikelompokkan
        st.markdown("#### 📋 **Detail Hasil Pengelompokan**")
        
        output_display = output.copy()
        output_display['Score'] = output_display['Score'].round(6)
        output_display.index.name = 'Ranking'
        
        st.dataframe(output_display, use_container_width=True)

        # Membuat diagram pie yang lebih menarik
        st.markdown("#### 🥧 **Visualisasi Distribusi Keringanan**")
        
        # Dapatkan warna untuk setiap kategori
        colors = [category_info.get(label, {}).get('color', '#gray') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.3,
            marker_colors=colors
        )])
        fig.update_layout(
            title='📊 Distribusi Mahasiswa per Kategori Keringanan UKT (Berdasarkan Threshold Skor)',
            annotations=[dict(text='Total<br>' + str(total), x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        fig.update_traces(textinfo='label+percent+value')
        st.plotly_chart(fig, use_container_width=True)

    # st.markdown(filedownload(output_final), unsafe_allow_html=True)

else:
    st.write("Mohon upload kedua file terlebih dahulu.")