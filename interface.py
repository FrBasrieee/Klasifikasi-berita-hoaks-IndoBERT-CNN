import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# KONFIGURASI
MODEL_PATH = "indobert_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Deteksi Berita Hoaks",
    layout="centered"
)

# CACHE MODEL & TOKENIZER
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "indobenchmark/indobert-base-p1"
    )

    class IndoBERT_CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.bert = AutoModel.from_pretrained(
                "indobenchmark/indobert-base-p1"
            )

            self.conv1 = nn.Conv1d(
                768, 128,
                kernel_size=3,
                padding=1
            )

            self.conv2 = nn.Conv1d(
                768, 128,
                kernel_size=4,
                padding=2
            )

            self.conv3 = nn.Conv1d(
                768, 128,
                kernel_size=5,
                padding=2
            )

            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(128 * 3, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            x = outputs.last_hidden_state
            x = x.permute(0, 2, 1)

            c1 = self.pool(self.relu(self.conv1(x))).squeeze(2)
            c2 = self.pool(self.relu(self.conv2(x))).squeeze(2)
            c3 = self.pool(self.relu(self.conv3(x))).squeeze(2)

            x = torch.cat([c1, c2, c3], dim=1)

            return self.fc(x)

    model = IndoBERT_CNN()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# STREAMLIT UI
st.title("📰 Deteksi Berita Hoaks Bahasa Indonesia")
st.write(
    "Masukkan **isi berita** untuk diklasifikasikan "
    "menggunakan model **IndoBERT + CNN**."
)

text = st.text_area(
    "Isi Berita",
    height=250,
    placeholder="Tempel isi berita di sini..."
)

if st.button("🔍 Deteksi"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        with st.spinner("Menganalisis berita..."):
            encoding = tokenizer(
                text[:3000],  # BATASI PANJANG TEKS (ANTI OOM)
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            label = "FAKTA" if pred == 0 else "HOAX"
            confidence = probs[0][pred].item() * 100

            st.success(f"Hasil Prediksi: **{label}**")
            st.write(f"Tingkat Kepercayaan: **{confidence:.2f}%**")

        # BERSIHKAN GPU SETELAH PREDIKSI
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

st.markdown("---")
st.caption(
    "Model: IndoBERT + CNN | "
    "Digunakan untuk keperluan akademik (Skripsi)"
)