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
    Pengecekan konsistensi untuk Triangular Fuzzy Numbers (TFN) using proper fuzzy methods
    
    Parameters:
    matrix: array of TFN - Matrix perbandingan berpasangan dalam format TFN [(l,m,u)]
    printComp: bool - Opsi untuk menampilkan detail perhitungan
    
    Returns:
    dict: Contains consistency results for lower, middle, upper bounds and overall assessment
    """
    mat_len = len(matrix)
    RI = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    # Ekstrak matrix untuk lower, middle, dan upper bounds
    lower_matrix = np.array([[tfn[0] for tfn in row] for row in matrix])
    middle_matrix = np.array([[tfn[1] for tfn in row] for row in matrix])
    upper_matrix = np.array([[tfn[2] for tfn in row] for row in matrix])
    
    if printComp:
        st.markdown("#### 🔢 **Dekomposisi Matrix TFN:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Lower Bound Matrix (L):**")
            st.write(pd.DataFrame(lower_matrix).round(3))
        
        with col2:
            st.write("**Middle Value Matrix (M):**")
            st.write(pd.DataFrame(middle_matrix).round(3))
        
        with col3:
            st.write("**Upper Bound Matrix (U):**")
            st.write(pd.DataFrame(upper_matrix).round(3))
    
    # Hitung konsistensi untuk setiap bound
    def calculate_consistency_for_matrix(matrix_bound, bound_name):
        eigenvalues = np.linalg.eigvals(matrix_bound)
        lambda_max = max(eigenvalues.real)  # Ambil bagian real dari eigenvalue
        
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
    
    # Hitung konsistensi untuk setiap bound
    lower_result = calculate_consistency_for_matrix(lower_matrix, "Lower")
    middle_result = calculate_consistency_for_matrix(middle_matrix, "Middle")
    upper_result = calculate_consistency_for_matrix(upper_matrix, "Upper")
    
    # Implementasi Geometric Consistency Index (GCI) untuk TFN
    def calculate_geometric_consistency():
        """Hitung GCI menggunakan geometric mean dari TFN"""
        gci_values = []
        
        for bound_matrix, bound_name in [(lower_matrix, "Lower"), (middle_matrix, "Middle"), (upper_matrix, "Upper")]:
            if mat_len <= 2:
                gci_values.append(0)
                continue
                
            # Hitung geometric mean untuk setiap baris
            row_gm = []
            for i in range(mat_len):
                product = 1
                for j in range(mat_len):
                    product *= bound_matrix[i][j]
                row_gm.append(product ** (1/mat_len))
            
            # Hitung GCI
            gci_sum = 0
            for i in range(mat_len):
                for j in range(mat_len):
                    if i != j:
                        gci_sum += (np.log(bound_matrix[i][j]) - np.log(row_gm[i]) + np.log(row_gm[j])) ** 2
            
            gci = (1 / (2 * (mat_len - 1) * (mat_len - 2))) * gci_sum if mat_len > 2 else 0
            gci_values.append(gci)
        
        return gci_values
    
    gci_values = calculate_geometric_consistency()
    
    # Fuzzy Consistency Assessment menggunakan defuzzification
    def fuzzy_defuzzification_cr():
        """Hitung CR menggunakan defuzzification (centroid method)"""
        # Defuzzify menggunakan centroid method: (l + m + u) / 3
        defuzz_matrix = np.zeros((mat_len, mat_len))
        for i in range(mat_len):
            for j in range(mat_len):
                l, m, u = matrix[i][j]
                defuzz_matrix[i][j] = (l + m + u) / 3
        
        eigenvalues = np.linalg.eigvals(defuzz_matrix)
        lambda_max = max(eigenvalues.real)
        
        ri_value = RI[mat_len] if mat_len in RI else RI[10] if mat_len >= 10 else 0
        ci_value = (lambda_max - mat_len) / (mat_len - 1) if mat_len > 1 else 0
        cr_value = ci_value / ri_value if ri_value > 0 else 0
        
        return {
            'lambda_max': lambda_max,
            'CI': ci_value,
            'RI': ri_value,
            'CR': cr_value,
            'consistent': cr_value <= 0.1
        }
    
    defuzz_result = fuzzy_defuzzification_cr()
    
    # Overall consistency assessment
    bounds_consistent = [lower_result['consistent'], middle_result['consistent'], upper_result['consistent']]
    overall_consistent = all(bounds_consistent) or defuzz_result['consistent']
    
    if printComp:
        st.markdown("#### 📊 **Hasil Analisis Konsistensi TFN:**")
        
        # Tabel hasil konsistensi
        consistency_df = pd.DataFrame({
            'Bound': ['Lower (L)', 'Middle (M)', 'Upper (U)', 'Defuzzified'],
            'λ_max': [lower_result['lambda_max'], middle_result['lambda_max'], 
                     upper_result['lambda_max'], defuzz_result['lambda_max']],
            'CI': [lower_result['CI'], middle_result['CI'], 
                   upper_result['CI'], defuzz_result['CI']],
            'RI': [lower_result['RI'], middle_result['RI'], 
                   upper_result['RI'], defuzz_result['RI']],
            'CR': [lower_result['CR'], middle_result['CR'], 
                   upper_result['CR'], defuzz_result['CR']],
            'Status': ['✅ Konsisten' if lower_result['consistent'] else '❌ Tidak Konsisten',
                      '✅ Konsisten' if middle_result['consistent'] else '❌ Tidak Konsisten',
                      '✅ Konsisten' if upper_result['consistent'] else '❌ Tidak Konsisten',
                      '✅ Konsisten' if defuzz_result['consistent'] else '❌ Tidak Konsisten']
        })
        
        st.dataframe(consistency_df.round(4), use_container_width=True)
        
        # GCI Results
        if any(gci > 0 for gci in gci_values):
            st.markdown("#### 🎯 **Geometric Consistency Index (GCI):**")
            gci_df = pd.DataFrame({
                'Bound': ['Lower (L)', 'Middle (M)', 'Upper (U)'],
                'GCI': gci_values,
                'Status': ['✅ Baik' if gci <= 0.31 else '❌ Perlu Revisi' for gci in gci_values]
            })
            st.dataframe(gci_df.round(4), use_container_width=True)
            st.info("💡 **Interpretasi GCI:** ≤ 0.31 = Konsistensi baik, > 0.31 = Perlu revisi")
        
        # Overall assessment
        st.markdown("#### 🏆 **Penilaian Konsistensi Keseluruhan:**")
        
        if overall_consistent:
            st.success("✅ **Matrix TFN KONSISTEN** - Dapat digunakan untuk perhitungan lanjutan")
        else:
            st.warning("⚠️ **Matrix TFN TIDAK KONSISTEN** - Disarankan untuk merevisi penilaian")
        
        # Detailed explanation
        with st.expander("🔍 **Penjelasan Detail Metode Konsistensi TFN**"):
            st.write("""
            **Metode yang Digunakan:**
            
            1. **Bound-wise Consistency:** Mengecek konsistensi pada setiap bound (L, M, U) secara terpisah
            2. **Defuzzification Consistency:** Menggunakan centroid method untuk defuzzifikasi TFN
            3. **Geometric Consistency Index (GCI):** Metode alternatif untuk penilaian konsistensi
            
            **Kriteria Penilaian:**
            - **CR ≤ 0.1:** Matrix konsisten
            - **GCI ≤ 0.31:** Konsistensi geometris baik
            - **Overall:** Matrix dianggap konsisten jika semua bound konsisten ATAU defuzzified CR ≤ 0.1
            
            **Kelebihan Metode TFN:**
            - Mempertimbangkan ketidakpastian dalam penilaian
            - Memberikan interval kepercayaan untuk konsistensi
            - Lebih robust terhadap variasi penilaian subjektif
            """)
    
    return {
        'lower': lower_result,
        'middle': middle_result,
        'upper': upper_result,
        'defuzzified': defuzz_result,
        'gci': gci_values,
        'overall_consistent': overall_consistent,
        'bounds_consistent': bounds_consistent
    }

def isConsistent(matrix, printComp=True):
    """
    Legacy function for backward compatibility - now uses fuzzy consistency check
    """
    result = fuzzy_consistency_check(matrix, printComp)
    return result['overall_consistent']

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
    if(printComp): 
        st.markdown("---")
        st.markdown("## 🔍 **TAHAP 1: PENGECEKAN KONSISTENSI MATRIKS**")
        
        with st.expander("ℹ️ Penjelasan Pengecekan Konsistensi TFN", expanded=False):
            st.write("""
            **Tujuan:** Memastikan bahwa matriks perbandingan berpasangan TFN konsisten dan dapat diandalkan.
            
            **Metode Konsistensi TFN yang Digunakan:**
            1. **Bound-wise Analysis:** Mengecek konsistensi pada Lower (L), Middle (M), dan Upper (U) bound secara terpisah
            2. **Defuzzification Method:** Menggunakan centroid method [(L+M+U)/3] untuk defuzzifikasi
            3. **Geometric Consistency Index (GCI):** Metode alternatif berbasis geometric mean
            
            **Kriteria Penilaian:**
            - **CR ≤ 0.1:** Matrix konsisten untuk setiap bound
            - **GCI ≤ 0.31:** Konsistensi geometris baik  
            - **Overall:** Konsisten jika semua bound konsisten ATAU defuzzified CR ≤ 0.1
            
            **Keunggulan Pendekatan TFN:**
            - Mempertimbangkan ketidakpastian dalam penilaian subjektif
            - Memberikan interval kepercayaan untuk hasil konsistensi
            - Lebih robust dan realistis untuk pengambilan keputusan fuzzy
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
    crxcr_weights = pairwiseComp(crxcr, printComp)
    
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
        altxalt_weights[i] =  pairwiseComp(altxalt_cr, printComp)

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