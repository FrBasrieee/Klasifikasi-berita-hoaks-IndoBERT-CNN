import pandas as pd
import re

# load hasil case folding
df = pd.read_excel("dataset/dataset_casefolding(2).xlsx")

# kamus normalisasi singkatan
kamus = {
    "yg": "yang",
    "jd": "jadi",
    "dr": "dari",
    "rmh": "rumah",
    "org": "orang",
    "vrl": "viral",
    "bpk": "bapak",
    "temen2": "teman teman",
    "pelan2": "pelan pelan",
    "hati2": "hati hati",
    "siap2": "siap  siap",
    "emak2": "emak emak",
    "adik2": "adik adik",
    "anak2": "anak anak",
    "saudara2": "saudara saudara",
    "keponakan2": "keponakan keponakan",
    "jangan2": "jangan jangan",
    "tiba2": "tiba tiba",
    "benar2": "benar benar",
    "pilih2": "pilih pilih",
    "pura2": "pura pura",
    "olok2": "olok olok",
    "shre": "share",
    "dlm": "dalam",
    "ass": "assalamualaikum",
    "pd": "pada",
    "kpd": "kepada",
    "msh": "masih",
    "knp": "kenapa",
    "bln": "bulan",
    "tgl": "tanggal"
    
}

def cleaning(text):
    text = str(text)
    
    # hapus URL
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # normalisasi singkatan
    words = text.split()
    words = [kamus[w] if w in kamus else w for w in words]
    text = " ".join(words)
    
    # hapus simbol aneh (simpan huruf angka spasi titik)
    text = re.sub(r'[^a-z0-9\s.]', ' ', text)
    
    # rapikan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# terapkan cleaning
df['Narasi'] = df['Narasi'].apply(cleaning)

# simpan hasil
df.to_excel("dataset/dataset_clean(3).xlsx", index=False)

print("Cleaning selesai")
print("Jumlah data:", len(df))