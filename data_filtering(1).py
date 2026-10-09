import pandas as pd

df = pd.read_excel("dataset/deteksi-berita-hoaks-indo-dataset.xlsx")

df['Narasi'] = df['Narasi'].astype(str)

# Hapus baris : kosong, "nan",""
df = df[
    df['Narasi'].notna() &
    (df['Narasi'].str.strip() != "") &
    (df['Narasi'].str.strip().str.lower() != "nan")
]

df.to_excel("dataset/dataset_filtering.xlsx", index=False)

print("Baris kosong berhasil dihapus")
print("Jumlah data sekarang:", len(df))