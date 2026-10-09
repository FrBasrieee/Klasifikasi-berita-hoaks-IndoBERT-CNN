import pandas as pd

df = pd.read_excel("dataset/dataset_tanpa_kosong(1).xlsx")

df['Narasi'] = df['Narasi'].astype(str)

# Case folding
df['Narasi'] = df['Narasi'].str.lower()

df.to_excel("dataset/dataset_casefolding(2).xlsx", index=False)

print("Case folding selesai")
print("Jumlah data:", len(df))
