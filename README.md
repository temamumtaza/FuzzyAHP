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

- Input kriteria dan alternatif untuk proses AHP
- Perhitungan matriks perbandingan berpasangan
- Validasi konsistensi (Consistency Ratio)
- Export hasil ke file Excel
- Visualisasi hasil dengan grafik interaktif

## Demo Aplikasi

Anda dapat mengakses aplikasi ini secara langsung di: https://fahp-ukt.streamlit.app

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
