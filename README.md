# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Selama lebih dari dua dekade beroperasi, institusi ini telah berhasil mencetak banyak lulusan dengan reputasi akademik yang baik di berbagai bidang studi, mulai dari agronomi, desain, pendidikan, keperawatan, jurnalisme, manajemen, layanan sosial, hingga teknologi.

Namun demikian, terdapat tantangan serius yang dihadapi: tingginya angka mahasiswa yang tidak menyelesaikan pendidikan mereka atau **dropout**. Dari total 4.424 mahasiswa dalam dataset, sebanyak **1.421 mahasiswa (32,1%)** tercatat sebagai dropout — angka yang cukup signifikan dan berpotensi merugikan reputasi serta keberlangsungan institusi.

### Permasalahan Bisnis

1. **Tingginya angka dropout mahasiswa** — Lebih dari sepertiga mahasiswa tidak berhasil menyelesaikan pendidikannya, berdampak langsung pada kualitas institusi, akreditasi, dan kepercayaan publik.

2. **Tidak adanya sistem deteksi dini risiko dropout** — Institusi belum memiliki mekanisme untuk mengidentifikasi mahasiswa berisiko dropout sedini mungkin, sehingga intervensi sering terlambat diberikan.

3. **Keterbatasan pemahaman terhadap faktor penyebab dropout** — Belum diketahui secara jelas faktor-faktor akademik, demografis, maupun sosial-ekonomi yang paling berpengaruh terhadap dropout.

4. **Minimnya alat monitoring performa mahasiswa** — Pihak manajemen belum memiliki dashboard yang memadai untuk memantau perkembangan mahasiswa secara menyeluruh.

**Business Questions:**
- Faktor apa yang paling signifikan memengaruhi dropout mahasiswa?
- Apakah faktor akademik lebih dominan dibanding faktor demografis atau sosial-ekonomi?
- Bagaimana profil tipikal mahasiswa yang berisiko tinggi dropout?
- Seberapa akurat model machine learning dapat memprediksi status mahasiswa?
- Apakah mahasiswa penerima beasiswa memiliki tingkat dropout lebih rendah?

### Cakupan Proyek

1. **Data Understanding** — Memahami struktur dataset, tipe variabel, distribusi nilai, serta identifikasi potensi masalah data.
2. **Exploratory Data Analysis (EDA)** — Analisis mendalam untuk memahami pola dropout dan hubungan antar variabel.
3. **Data Preparation / Preprocessing** — Encoding, feature engineering, outlier handling, normalisasi, train-test split, dan penanganan class imbalance (SMOTE).
4. **Modeling** — Benchmark model dengan LazyPredict, pemilihan top 5 model, dan hyperparameter tuning dengan RandomizedSearchCV.
5. **Evaluasi** — Mengukur performa model dengan accuracy, F1 score, ROC AUC, dan confusion matrix.
6. **Business Dashboard** — Dashboard interaktif di Metabase untuk monitoring performa mahasiswa.
7. **Prototype Machine Learning** — Aplikasi Streamlit untuk prediksi dropout secara individual maupun batch.
8. **Rekomendasi Action Items** — Langkah-langkah berbasis data untuk menurunkan angka dropout.

### Persiapan

**Sumber data:**
```
https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv
```

**Setup environment (Notebook):**
```bash
pip install pandas==2.2.2 numpy==2.0.2 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1 imbalanced-learn==0.14.1 xgboost==3.2.0 lightgbm==4.6.0 lazypredict==0.3.0 joblib==1.5.3
```

**Setup environment (Streamlit App):**
```bash
# Clone repository
git clone https://github.com/XMB234/Tugas-2-Penerapan-Data-Science.git
cd Tugas-2-Penerapan-Data-Science

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

---

## Business Dashboard

Dashboard **Jaya Jaya Institut — Student Performance Monitor** dibuat menggunakan **Metabase** dengan koneksi database SQLite. Dashboard ini dirancang untuk membantu pihak manajemen institusi memantau performa mahasiswa dan mengidentifikasi mahasiswa berisiko dropout secara visual dan interaktif.

### Fitur Dashboard

Dashboard terdiri dari 4 section utama:

**1. KPI (Key Performance Indicator)**
Menampilkan angka ringkasan utama:
- Total Mahasiswa: **4.424**
- Total Dropout: **1.421** (32.1%)
- Total Graduate: **2.209** (49.9%)
- High Risk Students: **1.152**
- Avg Risk Score: **31.7%**

**2. Overview Distribusi**
- Distribusi status mahasiswa (pie chart)
- Dropout rate per program studi — tertinggi: Biofuel Production Technologies (66.7%), Equinculture (55.3%), Informatics Engineering (54.1%)
- Tren distribusi status per jurusan (stacked bar chart)

**3. Analisis Faktor Risiko**
- Dropout rate by Gender: Male (45.1%) vs Female (25.1%)
- Dropout rate by Tuition Fee: Not Up to Date (86.6%) vs Up to Date (24.7%)
- Dropout rate by Debtor Status: Yes (62%) vs No (28.3%)
- Dropout rate by Attendance: Evening (42.9%) vs Daytime (30.8%)
- Dropout rate by Scholarship: No (38.7%) vs Yes (12.2%)
- Rata-rata nilai akademik per status (skala 0–20)

**4. Risk Monitoring**
- Distribusi Risk Category: Low Risk (65.89%), High Risk (26.04%), Medium Risk (8.07%)
- Risk category per status aktual
- Distribusi usia per status
- Tabel daftar mahasiswa High Risk (1.152 mahasiswa)

### Filter Dashboard
Dashboard dilengkapi dengan 6 filter interaktif: **Status**, **Risk Category**, **Program Studi**, **Gender**, **Beasiswa**, dan **Kelas** — memungkinkan pengamat mengeksplorasi data secara mendalam.

### Cara Mengakses Dashboard Metabase

**Prasyarat:** Install [Docker](https://www.docker.com/products/docker-desktop/) terlebih dahulu.

**Step 1 — Jalankan container Metabase:**
```bash
docker run -d -p 3000:3000 --name metabase metabase/metabase
```
Tunggu sekitar 1–2 menit hingga Metabase selesai starting up.

**Step 2 — Copy file database ke dalam container:**
```bash
# Copy konfigurasi dashboard (questions, chart, filter, koneksi)
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db

# Copy data mahasiswa (sumber data chart dashboard)
docker cp jaya_jaya_institut.db metabase:/tmp/jaya_jaya_institut.db
```

**Step 3 — Restart container agar konfigurasi ter-load:**
```bash
docker restart metabase
```

**Step 4 — Akses dashboard:**

Buka browser dan akses:
```
http://localhost:3000
```

Login dengan kredensial berikut:
```
Email    : root@mail.com
Password : root123
```

Setelah login, pilih dashboard **"Jaya Jaya Institut — Student Performance Monitor"** dari halaman utama Metabase.

> **Catatan:** File `metabase.db.mv.db` berisi seluruh konfigurasi dashboard (queries, charts, filter). File `jaya_jaya_institut.db` berisi data mahasiswa yang ditampilkan di setiap chart. Keduanya wajib ada agar dashboard dapat berjalan dengan benar.

---

## Menjalankan Sistem Machine Learning

Prototype sistem machine learning dibangun menggunakan **Streamlit** dengan model **RandomForestClassifier** (Accuracy: 76.95%).

### Fitur Aplikasi

1. **Prediksi Individual** — Form input data mahasiswa (demografis, akademik, keuangan) untuk memprediksi status dan risk score secara real-time
2. **Prediksi Batch (CSV)** — Upload file CSV berisi data banyak mahasiswa sekaligus, lengkap dengan download template dan hasil prediksi

Setiap prediksi menghasilkan:
- Status prediksi: Dropout / Enrolled / Graduate
- Risk Score Dropout (0–100%)
- Risk Category: Low / Medium / High Risk
- Rekomendasi tindakan berdasarkan hasil prediksi

### Cara Menjalankan Lokal

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Akses Prototype

🔗 **[https://tugas-2-penerapan-data-science-dpeaaaqsnswainy3qwcmub.streamlit.app/](https://tugas-2-penerapan-data-science-dpeaaaqsnswainy3qwcmub.streamlit.app/)**

---

## Conclusion

Proyek ini berhasil membangun sistem deteksi dini risiko dropout mahasiswa untuk Jaya Jaya Institut menggunakan pendekatan data science end-to-end.

### Jawaban atas Business Questions

**1. Faktor apa yang paling signifikan memengaruhi dropout?**
Faktor akademik (nilai dan jumlah mata kuliah yang disetujui) serta faktor keuangan (status SPP dan hutang) adalah prediktor terkuat dropout. Hal ini dikonfirmasi oleh analisis korelasi fitur dan feature importance dari model RandomForestClassifier.

**2. Apakah faktor akademik lebih dominan dibanding demografis/sosial-ekonomi?**
Ya. Faktor akademik dan keuangan terbukti jauh lebih dominan. Sementara faktor sosial-ekonomi makro seperti GDP, inflasi, dan unemployment rate tidak menunjukkan perbedaan signifikan antar kelompok status mahasiswa.

**3. Bagaimana profil tipikal mahasiswa berisiko tinggi dropout?**
Mahasiswa berisiko tinggi dropout umumnya memiliki karakteristik: berusia lebih tua saat mendaftar (±26 tahun), nilai akademik rendah di semester 1 & 2 (rata-rata di bawah 8/20), tidak membayar SPP tepat waktu, memiliki hutang, tidak menerima beasiswa, dan mengambil kelas malam.

**4. Seberapa akurat model memprediksi status mahasiswa?**
Model RandomForestClassifier mencapai Accuracy **76.95%**, F1 Score weighted **0.7662**, dan AUC Graduate **0.9315**. Performa ini cukup solid untuk digunakan sebagai sistem deteksi dini, terutama untuk kelas Graduate dan Dropout.

**5. Apakah penerima beasiswa memiliki dropout rate lebih rendah?**
Ya, terbukti signifikan. Mahasiswa penerima beasiswa hanya memiliki dropout rate **12.2%**, jauh lebih rendah dibanding non-penerima beasiswa sebesar **38.7%** — selisih lebih dari 3x lipat.

**6. Pada semester berapa tanda awal dropout mulai terlihat?**
Tanda-tanda awal dropout sudah terlihat sejak **Semester 1**. Mahasiswa dropout rata-rata hanya mendapat nilai 7.26/20 di semester 1, dan kondisinya memburuk di semester 2 menjadi 5.9/20. Sebaliknya, mahasiswa graduate konsisten di atas 12/20 di kedua semester. Pola penurunan nilai dari semester 1 ke semester 2 pada kelompok dropout merupakan sinyal peringatan dini yang sangat jelas — artinya **intervensi paling efektif adalah di semester 1**.

---

**Temuan Utama dari EDA:**
- **Faktor akademik** adalah prediktor terkuat dropout — mahasiswa dropout rata-rata hanya mendapat nilai 7.26/20 di semester 1 dan turun menjadi 5.9/20 di semester 2, jauh di bawah mahasiswa graduate (12.64/20 dan 12.7/20)
- **Faktor keuangan** sangat kritis — mahasiswa yang tidak membayar SPP tepat waktu memiliki dropout rate 86.6%, sedangkan mahasiswa dengan hutang mencapai 62%
- **Penerima beasiswa** memiliki dropout rate hanya 12.2%, jauh lebih rendah dibanding non-beasiswa (38.7%)
- **Mahasiswa kelas malam** lebih rentan dropout (42.9%) dibanding kelas siang (30.8%)
- **Usia pendaftaran** berpengaruh — mahasiswa dropout rata-rata berusia 26 tahun, lebih tua dibanding enrolled (22 tahun) dan graduate (22 tahun)
- **Faktor sosial-ekonomi makro** (GDP, inflasi, unemployment) tidak menunjukkan perbedaan signifikan antar kelompok status

**Performa Model:**
Model terbaik **RandomForestClassifier** (setelah hyperparameter tuning dengan RandomizedSearchCV, 50 iterasi, 3-fold CV) menghasilkan:
- Accuracy: **76.95%**
- F1 Score (weighted): **0.7662**
- AUC Graduate: **0.9315** (sangat baik)
- AUC Dropout: baik, dengan recall yang solid untuk deteksi dini

### Rekomendasi Action Items

1. **Intervensi Keuangan Prioritas Tinggi** — Identifikasi mahasiswa dengan status SPP tidak lancar atau memiliki hutang sejak semester pertama. Tawarkan skema cicilan, keringanan, atau informasikan program beasiswa yang tersedia. Penerima beasiswa terbukti memiliki dropout rate 3x lebih rendah.

2. **Program Monitoring Akademik Semester 1** — Mahasiswa dengan nilai di bawah 8/20 atau jumlah mata kuliah yang disetujui rendah di semester 1 harus segera mendapat bimbingan khusus dari dosen wali, karena tren nilai yang buruk di semester 1 hampir selalu berlanjut di semester 2.

3. **Perhatian Khusus pada Jurusan Berisiko Tinggi** — Program studi Biofuel Production (66.7%), Equinculture (55.3%), dan Informatics Engineering (54.1%) memiliki dropout rate tertinggi. Lakukan evaluasi kurikulum, beban studi, dan dukungan akademik di jurusan-jurusan ini.

4. **Program Dukungan Mahasiswa Kelas Malam** — Mahasiswa kelas malam kemungkinan besar bekerja sambil kuliah. Sediakan fasilitas konseling, jadwal yang fleksibel, dan dukungan khusus untuk kelompok ini guna menurunkan dropout rate dari 42.9%.

5. **Implementasi Sistem Early Warning berbasis Model ML** — Gunakan prototype Streamlit yang telah dibangun untuk secara rutin (tiap awal semester) memprediksi mahasiswa High Risk. Mahasiswa dengan Risk Score > 60% harus diprioritaskan untuk mendapat intervensi langsung dari tim akademik.

6. **Program Pendampingan Mahasiswa Berusia Lebih Tua** — Mahasiswa yang mendaftar di atas usia 25 tahun cenderung memiliki tantangan berbeda (pekerjaan, keluarga). Sediakan program mentoring khusus dan opsi pembelajaran yang lebih fleksibel untuk kelompok ini.
