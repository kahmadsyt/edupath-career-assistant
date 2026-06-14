import streamlit as st
from pathlib import Path
import yaml


# =========================
# Load Configuration
# =========================
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def load_config():
    """Load project configuration from config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return {}


config = load_config()

page_title = config.get("streamlit", {}).get("page_title", "EduPath Career Assistant")
page_icon = config.get("streamlit", {}).get("page_icon", "🎓")
layout = config.get("streamlit", {}).get("layout", "wide")


# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(
    page_title=page_title,
    page_icon=page_icon,
    layout=layout
)


# =========================
# App Header
# =========================
st.title("🎓 EduPath Career Assistant")
st.subheader("Prototype Chatbot Rekomendasi Program Studi S1 dan Roadmap Belajar")

st.markdown(
    """
    Aplikasi ini merupakan prototype awal untuk project Data Mining S2.

    Pada tahap selanjutnya, sistem akan dikembangkan untuk:
    - Melakukan intent classification
    - Memberikan rekomendasi Top-3/Top-5 program studi S1
    - Menjelaskan alasan rekomendasi
    - Menyusun roadmap belajar awal
    - Menampilkan prospek karier
    """
)


# =========================
# User Input Area
# =========================
st.markdown("### Masukkan Profil Minat Siswa")

user_input = st.text_area(
    "Contoh: Saya suka matematika, tertarik dengan komputer, senang menganalisis data, dan ingin bekerja di bidang teknologi.",
    height=150
)

top_k = st.selectbox(
    "Jumlah rekomendasi program studi:",
    options=[3, 5],
    index=1
)


# =========================
# Temporary Button Logic
# =========================
if st.button("Dapatkan Rekomendasi"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan deskripsi minat terlebih dahulu.")
    else:
        st.info("Tahap 01 berhasil. Model rekomendasi akan dikembangkan pada tahap berikutnya.")

        st.markdown("### Input yang diterima:")
        st.write(user_input)

        st.markdown("### Status Sistem:")
        st.success(
            f"Prototype berjalan normal. Nantinya sistem akan menampilkan Top-{top_k} rekomendasi program studi."
        )


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("EduPath Career Assistant | Data Mining S2 | Tahap 01")