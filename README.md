# Eksperimen SML - Muhammad Anugrah

Repository ini dibuat untuk Kriteria 1 submission Membangun Sistem Machine Learning.

Struktur repository:

```text
Eksperimen_SML_Muhammad_Anugrah
├── .github/workflows/preprocessing.yml
├── .workflow/preprocessing.yml
├── breast_cancer_raw
│   └── breast_cancer_raw.csv
├── preprocessing
│   ├── Eksperimen_Muhammad_Anugrah.ipynb
│   ├── automate_Muhammad_Anugrah.py
│   └── breast_cancer_preprocessing
│       ├── breast_cancer_preprocessed.csv
│       ├── train.csv
│       └── test.csv
├── README.md
└── requirements.txt
```

Notebook `preprocessing/Eksperimen_Muhammad_Anugrah.ipynb` mengikuti Template Eksperimen MSML dan berisi tahap:

1. Perkenalan Dataset
2. Import Library
3. Memuat Dataset
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing

Script otomatisasi preprocessing dapat dijalankan dengan:

```bash
pip install -r requirements.txt
python preprocessing/automate_Muhammad_Anugrah.py
```

Workflow GitHub Actions akan menjalankan preprocessing otomatis saat ada push ke branch `main` atau saat dijalankan manual melalui `workflow_dispatch`.
