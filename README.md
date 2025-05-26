# Seleksi Keringanan UKT dengan Metode AHP

Aplikasi ini dibuat untuk membantu proses seleksi keringanan UKT menggunakan metode AHP (Analytic Hierarchy Process).

## Persyaratan Sistem

- Python 3.9 atau lebih tinggi
- pip (Python package installer)

## Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd FuzzyAHP
```

### 2. Install Dependencies
Jalankan perintah berikut untuk menginstall semua dependencies yang diperlukan:

```bash
pip install -r requirements.txt
```

Dependencies yang akan diinstall:
- `streamlit` - Framework untuk membuat aplikasi web
- `pandas` - Library untuk manipulasi data
- `numpy` - Library untuk komputasi numerik
- `xlsxwriter` - Library untuk menulis file Excel
- `plotly` - Library untuk visualisasi interaktif
- `openpyxl` - Library untuk membaca/menulis file Excel

### 3. Verifikasi Instalasi
Pastikan semua dependencies terinstall dengan benar:
```bash
pip list | grep -E "(streamlit|pandas|numpy|xlsxwriter|plotly|openpyxl)"
```

## Menjalankan Aplikasi

Setelah instalasi selesai, jalankan aplikasi dengan perintah:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan di browser pada alamat: `http://localhost:8501`

## Fitur Aplikasi

### 🔍 Fuzzy AHP dengan Konsistensi TFN yang Robust

Aplikasi ini mengimplementasikan pendekatan konsistensi yang proper untuk Triangular Fuzzy Numbers (TFN):

#### **Metode Konsistensi TFN:**
1. **Bound-wise Analysis** - Mengecek konsistensi pada setiap bound (Lower, Middle, Upper) secara terpisah
2. **Defuzzification Method** - Menggunakan centroid method [(L+M+U)/3] untuk defuzzifikasi
3. **Geometric Consistency Index (GCI)** - Metode alternatif berbasis geometric mean

#### **Keunggulan Pendekatan TFN:**
- ✅ Mempertimbangkan ketidakpastian dalam penilaian subjektif
- ✅ Memberikan interval kepercayaan untuk hasil konsistensi  
- ✅ Lebih robust terhadap variasi penilaian
- ✅ Analisis komprehensif dengan multiple bounds
- ✅ Implementasi standar akademik untuk Fuzzy AHP

### 📊 Fitur Utama Aplikasi:

- Input kriteria dan alternatif untuk proses AHP
- Perhitungan matriks perbandingan berpasangan
- **Tampilan Matrix Pairwise Kriteria** - Menampilkan matrix perbandingan berpasangan kriteria x kriteria dengan format Triangular Fuzzy Numbers (TFN)
- **Pengecekan Konsistensi TFN yang Komprehensif:**
  - Analisis per-bound (L, M, U)
  - Defuzzification consistency
  - Geometric Consistency Index (GCI)
  - Status konsistensi keseluruhan
- Toggle untuk menampilkan/menyembunyikan detail perhitungan
- Validasi konsistensi dengan multiple methods
- Pengelompokan hasil berdasarkan:
  - Alokasi persentase (dengan slider untuk mengatur kuota)
  - Batas skor (dengan slider untuk mengatur threshold)
- Visualisasi hasil dengan grafik pie chart interaktif
- Export hasil ke file Excel

### 🎯 Kriteria Penilaian Konsistensi:
- **CR ≤ 0.1:** Matrix konsisten untuk setiap bound
- **GCI ≤ 0.31:** Konsistensi geometris baik
- **Overall Consistency:** Matrix dianggap konsisten jika semua bound konsisten ATAU defuzzified CR ≤ 0.1

## Demo Aplikasi

Anda dapat mengakses aplikasi ini secara langsung di: https://fahp-ukt.streamlit.app

## Metodologi TFN Consistency

### Background
Dalam Fuzzy AHP tradisional, pengecekan konsistensi sering hanya menggunakan nilai tengah (middle value) dari TFN, yang tidak optimal karena:
- Tidak memanfaatkan informasi interval ketidakpastian
- Mengabaikan kemungkinan inkonsistensi pada bound tertentu
- Kurang robust untuk decision making yang melibatkan uncertainty

### Solusi yang Diimplementasikan
Aplikasi ini menggunakan pendekatan multi-faceted untuk konsistensi TFN:

1. **Decomposition Approach**: Menganalisis setiap bound secara terpisah
2. **Defuzzification Approach**: Menggunakan centroid method untuk representasi tunggal
3. **Geometric Approach**: Implementasi GCI untuk validasi tambahan

Hal ini memberikan confidence interval dan robustness yang lebih baik dalam pengambilan keputusan.

## Troubleshooting

### Error: ModuleNotFoundError
Jika muncul error module tidak ditemukan, pastikan Anda sudah menginstall dependencies:
```bash
pip install -r requirements.txt
```

### Error: Permission Denied
Jika menggunakan macOS/Linux dan mendapat error permission, gunakan:
```bash
pip install --user -r requirements.txt
```

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan fork repository dan buat pull request.

---

Terima kasih dan semoga bermanfaat!
