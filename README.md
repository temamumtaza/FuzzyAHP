# 🎯 Fuzzy AHP untuk Seleksi Keringanan UKT

<div align="center">

![Fuzzy AHP](https://img.shields.io/badge/Fuzzy-AHP-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Sistem Pendukung Keputusan Berbasis Fuzzy AHP untuk Seleksi Keringanan UKT**  
*Decision Support System with Advanced Triangular Fuzzy Numbers (TFN) Consistency Checking*

[🚀 Live Demo](https://fahp-ukt.streamlit.app) • [📖 Dokumentasi](#dokumentasi) • [💡 Fitur](#fitur-utama) • [🛠️ Instalasi](#instalasi)

</div>

---

## 📋 Deskripsi

Aplikasi **Fuzzy AHP untuk Seleksi Keringanan UKT** adalah sistem pendukung keputusan berbasis web yang dirancang untuk membantu institusi pendidikan dalam proses seleksi penerima keringanan Uang Kuliah Tunggal (UKT). Aplikasi ini menggunakan metode **Fuzzy Analytic Hierarchy Process (FAHP)** dengan implementasi **Alpha-Cut Consistency Checking** yang robust dan sesuai standar akademik.

### 🎯 Tujuan Aplikasi
- Memberikan objektivitas dalam proses seleksi keringanan UKT
- Mengurangi bias subjektif dalam pengambilan keputusan
- Menyediakan transparansi dan akuntabilitas dalam proses seleksi
- Mengoptimalkan alokasi bantuan berdasarkan kriteria yang terukur

---

## ✨ Fitur Utama

### 🔍 **Advanced Fuzzy AHP Implementation**
- **Alpha-Cut Consistency Method** - Transformasi TFN ke crisp matrix dengan α=1
- **Triangular Fuzzy Numbers (TFN)** - Representasi ketidakpastian dalam penilaian
- **Robust Consistency Checking** - Validasi matrix dengan Consistency Ratio (CR ≤ 0.1)
- **Automated Matrix Generation** - Pembentukan matrix pairwise otomatis dari data input

### 📊 **Interactive Data Visualization**
- **Matrix Pairwise Display** - Tampilan matrix perbandingan kriteria dengan format TFN
- **Dynamic Pie Charts** - Visualisasi distribusi bobot kriteria dan hasil pengelompokan
- **Responsive Tables** - Tabel interaktif dengan format yang mudah dibaca
- **Progress Indicators** - Status konsistensi dengan indikator visual

### ⚙️ **Flexible Grouping System**
- **Percentage Allocation** - Pengelompokan berdasarkan kuota dan persentase
- **Score Threshold** - Pengelompokan berdasarkan batas skor minimum
- **Interactive Sliders** - Pengaturan parameter secara real-time
- **Multiple Categories** - Keringanan 50%, 30%, 20%, dan tanpa keringanan

### 🎨 **Modern User Interface**
- **Professional Design** - Interface yang clean dan user-friendly
- **Collapsible Sections** - Organisasi konten dengan expandable sections
- **Responsive Layout** - Optimized untuk desktop dan mobile
- **Educational Content** - Penjelasan detail untuk setiap tahap perhitungan

---

## 🛠️ Instalasi

### Persyaratan Sistem
- **Python**: 3.9 atau lebih tinggi
- **pip**: Python package installer
- **Browser**: Chrome, Firefox, Safari, atau Edge (terbaru)

### Langkah Instalasi

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/temamumtaza/FuzzyAHP.git
cd FuzzyAHP
```

#### 2️⃣ Buat Virtual Environment (Opsional tapi Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies yang akan diinstall:**
| Package | Version | Deskripsi |
|---------|---------|-----------|
| `streamlit` | Latest | Framework untuk aplikasi web |
| `pandas` | Latest | Manipulasi dan analisis data |
| `numpy` | Latest | Komputasi numerik |
| `plotly` | Latest | Visualisasi interaktif |
| `openpyxl` | Latest | Pembaca/penulis file Excel |
| `xlsxwriter` | Latest | Penulis file Excel lanjutan |

#### 4️⃣ Verifikasi Instalasi
```bash
pip list | grep -E "(streamlit|pandas|numpy|plotly|openpyxl|xlsxwriter)"
```

---

## 🚀 Menjalankan Aplikasi

### Metode 1: Local Development
```bash
streamlit run streamlit_app.py
```

### Metode 2: Dengan Port Khusus
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Metode 3: Headless Mode (untuk Server)
```bash
streamlit run streamlit_app.py --server.headless true
```

**Akses Aplikasi:**
- **Local**: http://localhost:8501
- **Network**: http://[your-ip]:8501

### 🌐 Live Demo
Aplikasi juga tersedia online di: **[https://fahp-ukt.streamlit.app](https://fahp-ukt.streamlit.app)**

---

## 📖 Cara Penggunaan

### 1️⃣ **Persiapan Data**
- Download template file Excel dari folder [Data Sample](https://github.com/temamumtaza/FuzzyAHP/tree/main/Data%20Sample)
- **File Kriteria**: Berisi daftar kriteria dan nilai importance
- **File Alternatif**: Berisi daftar mahasiswa dan nilai untuk setiap kriteria

### 2️⃣ **Upload File**
- Upload file kriteria (.xlsx)
- Upload file alternatif (.xlsx)
- Pastikan format sesuai dengan template

### 3️⃣ **Konfigurasi Tampilan**
- ✅ Centang: "🔍 Tampilkan Detail Perhitungan Fuzzy AHP" untuk melihat:
  - Matrix pairwise kriteria
  - Detail perhitungan konsistensi
  - Fuzzy geometric mean values
  - Proses perhitungan weights

### 4️⃣ **Pilih Metode Pengelompokan**
- **Alokasi Persentase**: Berdasarkan kuota dan distribusi persen
- **Batas Skor**: Berdasarkan threshold skor minimum

### 5️⃣ **Analisis Hasil**
- Review ranking alternatif
- Analisis distribusi pengelompokan
- Download hasil (jika diperlukan)

---

## 📊 Metodologi

### 🔢 **Alpha-Cut Consistency Method**
Aplikasi menggunakan metode Alpha-Cut dengan α=1 untuk pengecekan konsistensi:

1. **Transformasi TFN → Crisp**: `TFN(l,m,u) → m` (nilai tengah)
2. **Perhitungan Eigenvalue**: Menggunakan crisp matrix
3. **Consistency Index**: `CI = (λmax - n) / (n - 1)`
4. **Consistency Ratio**: `CR = CI / Random Index`
5. **Kriteria Konsistensi**: `CR ≤ 0.1`

### 🎯 **Logic TFN yang Digunakan**
| Selisih Nilai | TFN | Interpretasi |
|---------------|-----|--------------|
| i = j atau sama | (1,1,3) | Sama penting atau diagonal |
| diff = 1 | (1,3,5) | Sedikit lebih penting |
| diff = 2 | (3,5,7) | Lebih penting |
| diff = 3 | (5,7,9) | Sangat penting |
| diff ≥ 4 | (7,9,9) | Mutlak lebih penting |

### 📈 **Keunggulan Implementasi**
- ✅ **Robust**: Menggunakan metode yang telah teruji secara akademik
- ✅ **Efficient**: Komputasi yang optimal untuk dataset besar
- ✅ **Deterministic**: Hasil yang konsisten dan dapat direproduksi
- ✅ **Transparent**: Setiap tahap perhitungan dapat di-trace

---

## 🔧 Troubleshooting

### ❌ **Error: ModuleNotFoundError**
```bash
# Solusi: Install ulang dependencies
pip install -r requirements.txt

# Jika masih error, gunakan virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# atau
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### ❌ **Error: Permission Denied**
```bash
# macOS/Linux
pip install --user -r requirements.txt

# Atau gunakan sudo (tidak disarankan)
sudo pip install -r requirements.txt
```

### ❌ **Port Already in Use**
```bash
# Gunakan port lain
streamlit run streamlit_app.py --server.port 8502

# Atau kill process yang menggunakan port
lsof -ti:8501 | xargs kill -9  # macOS/Linux
```

### ❌ **File Upload Error**
- Pastikan file Excel dalam format `.xlsx`
- Periksa struktur file sesuai template
- File size maksimal 200MB

---

## 🤝 Kontribusi

Kami menyambut kontribusi dari komunitas! Berikut cara berkontribusi:

### 🌟 **Jenis Kontribusi**
- 🐛 **Bug Reports**: Laporkan bug yang ditemukan
- 💡 **Feature Requests**: Usulan fitur baru
- 📖 **Documentation**: Perbaikan dokumentasi
- 🔧 **Code Contributions**: Kontribusi kode

### 📝 **Langkah Kontribusi**
1. Fork repository ini
2. Buat branch baru: `git checkout -b feature/nama-fitur`
3. Commit perubahan: `git commit -m 'Add some feature'`
4. Push ke branch: `git push origin feature/nama-fitur`
5. Buat Pull Request

---

## 📄 Lisensi

Proyek ini dilisensikan under [MIT License](LICENSE) - lihat file LICENSE untuk detail.

---

## 👥 Tim Pengembang

<div align="center">

**Dikembangkan dengan ❤️ oleh Tim Fuzzy AHP**

[![GitHub](https://img.shields.io/badge/GitHub-temamumtaza-181717?style=for-the-badge&logo=github)](https://github.com/temamumtaza)

</div>

---

## 📞 Dukungan

Jika Anda membutuhkan bantuan atau memiliki pertanyaan:

- 📧 **Email**: [Buat Issue di GitHub](https://github.com/temamumtaza/FuzzyAHP/issues)
- 💬 **Diskusi**: [GitHub Discussions](https://github.com/temamumtaza/FuzzyAHP/discussions)
- 📚 **Dokumentasi**: [Wiki](https://github.com/temamumtaza/FuzzyAHP/wiki)

---

<div align="center">

**⭐ Jika aplikasi ini bermanfaat, jangan lupa berikan star di GitHub! ⭐**

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=temamumtaza.FuzzyAHP)

</div>
