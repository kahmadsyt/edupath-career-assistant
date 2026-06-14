"""
Streamlit application for EduPath Career Assistant.

Run locally from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from chatbot_utils import EduPathChatbot


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="EduPath Career Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }
    .program-card {
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1.1rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        margin-bottom: 0.9rem;
    }
    .small-note {
        font-size: 0.88rem;
        color: #6b7280;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        font-size: 0.82rem;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# Cached Resource Loader
# ============================================================

@st.cache_resource(show_spinner="Memuat dataset, model intent classification, dan recommender system...")
def load_chatbot() -> EduPathChatbot:
    return EduPathChatbot()


def load_stage07_metrics(project_root: Path) -> pd.DataFrame:
    metrics_path = project_root / "reports" / "stage07" / "chatbot_evaluation_metrics_stage07.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def add_message(role: str, content: str, result: Dict[str, Any] | None = None) -> None:
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "result": result,
    })


# ============================================================
# Render Functions
# ============================================================


def render_metadata(result: Dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Final Intent", str(result.get("final_intent", "-")))
    cols[1].metric("Routing Source", str(result.get("routing_source", "-")))
    cols[2].metric("Model Intent", str(result.get("model_predicted_intent", "-")))
    cols[3].metric("Confidence", str(result.get("model_confidence", "-")))

    matched_keywords = result.get("matched_rule_keywords", "")
    processed_text = result.get("processed_text", "")

    if matched_keywords or processed_text:
        with st.expander("Detail routing dan preprocessing"):
            st.write("**Matched rule keywords:**", matched_keywords if matched_keywords else "-")
            st.write("**Processed text:**", processed_text if processed_text else "-")

            top_intents = result.get("top_intents", [])
            if top_intents:
                st.dataframe(pd.DataFrame(top_intents), use_container_width=True, hide_index=True)


def render_recommendation_cards(payload: pd.DataFrame) -> None:
    for _, row in payload.iterrows():
        rank = int(row.get("rank", 0))
        name = row.get("nama_program_studi", "-")
        jenjang = row.get("jenjang", "-")
        final_score = row.get("final_score", "-")
        bidang = row.get("bidang_keilmuan", "-")
        description = row.get("deskripsi_singkat", "-")
        keywords = row.get("matched_keywords", "-")
        careers = row.get("prospek_karier", "-")
        skills = row.get("skill_awal", "-")

        st.markdown(
            f"""
            <div class="program-card">
                <h4>#{rank} — {name} ({jenjang})</h4>
                <p><b>Bidang keilmuan:</b> {bidang}</p>
                <p><b>Deskripsi:</b> {description}</p>
                <p><b>Alasan rekomendasi:</b> cocok dengan keyword <code>{keywords}</code>.</p>
                <p><b>Skor akhir:</b> {final_score}</p>
                <p><b>Prospek karier:</b> {careers}</p>
                <p><b>Skill awal:</b> {skills}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_roadmap(payload: pd.DataFrame) -> None:
    for _, row in payload.iterrows():
        title = f"Fase {row.get('urutan_fase', '-')} — {row.get('fase', '-')} ({row.get('durasi_rekomendasi', '-')})"
        with st.expander(title, expanded=True):
            st.markdown(f"**Tujuan:** {row.get('tujuan_fase', '-')}")
            st.markdown(f"**Materi pokok:** {row.get('materi_pokok', '-')}")
            st.markdown(f"**Aktivitas praktik:** {row.get('aktivitas_praktik', '-')}")
            st.markdown(f"**Output portofolio:** {row.get('output_portofolio', '-')}")
            st.markdown(f"**Tools:** {row.get('tools_umum', '-')}")


def render_payload(result: Dict[str, Any], bot: EduPathChatbot | None = None) -> None:
    payload = result.get("payload")
    if not isinstance(payload, pd.DataFrame) or payload.empty:
        return

    final_intent = result.get("final_intent")

    if final_intent == "rekomendasi_prodi" and "nama_program_studi" in payload.columns:
        st.markdown("#### Ringkasan Rekomendasi")
        render_recommendation_cards(payload)
        if bot is not None and "program_id" in payload.columns:
            top_program_id = payload.iloc[0].get("program_id")
            top_program_name = payload.iloc[0].get("nama_program_studi", "program studi teratas")
            roadmap_df = bot.get_roadmap_by_program_id(str(top_program_id))
            if not roadmap_df.empty:
                st.markdown(f"#### Roadmap Belajar untuk Rekomendasi Teratas: {top_program_name}")
                render_roadmap(roadmap_df)

        with st.expander("Lihat tabel skor rekomendasi"):
            selected_cols = [
                col for col in [
                    "rank", "nama_program_studi", "jenjang", "final_score",
                    "text_similarity", "structured_score", "matched_keywords",
                    "prospek_karier", "skill_awal"
                ] if col in payload.columns
            ]
            st.dataframe(payload[selected_cols], use_container_width=True, hide_index=True)

    elif final_intent == "roadmap_belajar" and "fase" in payload.columns:
        st.markdown("#### Roadmap Belajar")
        render_roadmap(payload)

    else:
        with st.expander("Lihat data pendukung"):
            st.dataframe(payload, use_container_width=True, hide_index=True)


def render_result(result: Dict[str, Any], bot: EduPathChatbot | None = None, show_debug: bool = False) -> None:
    st.markdown(result.get("response_text", "-"))
    render_payload(result, bot=bot)

    if show_debug:
        st.divider()
        render_metadata(result)


def build_profile_prompt(
    jenjang: str,
    kelas: str,
    mapel: str,
    minat: str,
    hobi: str,
    gaya_belajar: str,
    tujuan_karier: str,
) -> str:
    return (
        f"Saya adalah siswa {jenjang} kelas {kelas}. "
        f"Mata pelajaran favorit saya adalah {mapel}. "
        f"Minat saya adalah {minat}. "
        f"Hobi atau aktivitas yang sering saya lakukan adalah {hobi}. "
        f"Gaya belajar saya cenderung {gaya_belajar}. "
        f"Tujuan karier saya adalah {tujuan_karier}. "
        "Program studi S1 apa yang cocok untuk saya? Berikan rekomendasi program studi beserta alasan."
    )


def process_user_input(bot: EduPathChatbot, text: str, top_n: int) -> None:
    user_text = str(text).strip()
    if not user_text:
        st.warning("Mohon isi pertanyaan atau profil siswa terlebih dahulu.")
        return

    add_message("user", user_text)

    try:
        result = bot.chatbot_response(user_text, top_n=top_n, return_dict=True)
    except Exception as exc:  # noqa: BLE001 - Streamlit app should show user-friendly diagnostics.
        result = {
            "response_text": (
                "Maaf, terjadi kendala saat memproses input. "
                "Silakan periksa kembali file dataset, model artifact, dan struktur folder project."
            ),
            "final_intent": "runtime_error",
            "routing_source": "exception_handler",
            "model_predicted_intent": "-",
            "model_confidence": 0.0,
            "processed_text": "",
            "matched_rule_keywords": "",
            "payload": pd.DataFrame(),
            "error_detail": str(exc),
        }

    add_message("assistant", result.get("response_text", "-"), result=result)


# ============================================================
# Main Application
# ============================================================

init_session_state()

try:
    bot = load_chatbot()
    system_summary = bot.get_system_summary()
    project_root = Path(system_summary["project_root"])
except Exception as exc:  # noqa: BLE001
    st.error("Aplikasi gagal dimuat.")
    st.markdown(
        "Pastikan folder `data/processed/` dan `models/` sudah tersedia pada root project, "
        "serta semua artifact dari Tahap 04–05 sudah tersimpan dengan nama file yang benar."
    )
    with st.expander("Detail error"):
        st.exception(exc)
    st.stop()

with st.sidebar:
    st.markdown("## 🎓 EduPath")
    st.caption("Career Assistant berbasis NLP, Intent Classification, dan Recommender System")

    top_n = st.slider("Jumlah rekomendasi program studi", min_value=1, max_value=5, value=3, step=1)
    show_debug = st.toggle("Tampilkan detail teknis", value=False)

    st.divider()
    st.markdown("### Status Sistem")
    st.success("Dataset dan model berhasil dimuat.")
    st.write("**Model intent:**", system_summary.get("best_model_name", "-"))
    st.write("**Metode recsys:** Content-Based Filtering")

    with st.expander("Ringkasan dataset"):
        dataset_rows = system_summary.get("dataset_rows", {})
        summary_df = pd.DataFrame(
            [{"dataset": key, "jumlah_baris": value} for key, value in dataset_rows.items()]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if st.button("Reset percakapan", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### Contoh Pertanyaan")
    st.markdown(
        """
        - Saya suka matematika dan data, jurusan apa yang cocok?
        - Aku seneng ngoding dan bikin aplikasi, cocok kuliah apa?
        - Kalau masuk Sains Data harus belajar apa dulu?
        - Prospek kerja Sistem Informasi apa saja?
        - Skill awal untuk UI UX Designer apa saja?
        """
    )

st.markdown('<div class="main-title">EduPath Career Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Chatbot rekomendasi program studi S1 dan roadmap belajar untuk siswa SMA/SMK/MA berbasis NLP, intent classification, dan recommender system.</div>',
    unsafe_allow_html=True,
)

main_tab, model_tab, guide_tab = st.tabs([
    "💬 Chatbot",
    "📊 Dataset & Model",
    "📘 Panduan",
])

with main_tab:
    st.markdown("### Input Profil Siswa SMA/SMK/MA")
    st.caption("Isi profil singkat berikut untuk memperoleh rekomendasi program studi yang lebih terarah.")

    with st.form("student_profile_form"):
        col1, col2, col3 = st.columns(3)
        jenjang = col1.selectbox("Jenjang", ["SMA", "SMK", "MA"])
        kelas = col2.selectbox("Kelas", ["10", "11", "12"])
        gaya_belajar = col3.selectbox(
            "Gaya belajar dominan",
            ["praktik", "logis dan analitis", "visual", "membaca teori", "diskusi", "eksperimen"],
        )

        mapel = st.text_input("Mata pelajaran favorit", placeholder="Contoh: Matematika, Informatika, Ekonomi")
        minat = st.text_input("Minat utama", placeholder="Contoh: data, coding, bisnis, desain, komunikasi")
        hobi = st.text_input("Hobi atau aktivitas yang disukai", placeholder="Contoh: membuat aplikasi, menganalisis data, desain poster")
        tujuan_karier = st.text_input("Tujuan karier", placeholder="Contoh: Data Scientist, Software Engineer, UI/UX Designer")

        submitted_profile = st.form_submit_button("Dapatkan Rekomendasi dari Profil", use_container_width=True)

    if submitted_profile:
        profile_prompt = build_profile_prompt(
            jenjang=jenjang,
            kelas=kelas,
            mapel=mapel,
            minat=minat,
            hobi=hobi,
            gaya_belajar=gaya_belajar,
            tujuan_karier=tujuan_karier,
        )
        process_user_input(bot, profile_prompt, top_n=top_n)
        st.rerun()

    st.divider()
    st.markdown("### Percakapan Chatbot")

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Halo, saya **EduPath Career Assistant**. Ceritakan minat, mapel favorit, hobi, gaya belajar, "
                "atau tujuan karier Anda. Saya akan memberikan rekomendasi program studi S1, alasan, roadmap belajar, dan prospek karier."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("result"):
                render_result(message["result"], bot=bot, show_debug=show_debug)
                if message["result"].get("error_detail") and show_debug:
                    st.error(message["result"]["error_detail"])
            else:
                st.markdown(message["content"])

    user_prompt = st.chat_input("Tulis pertanyaan Anda di sini...")
    if user_prompt:
        process_user_input(bot, user_prompt, top_n=top_n)
        st.rerun()

with model_tab:
    st.markdown("### Ringkasan Dataset dan Artifact Model")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Intent Data", system_summary["dataset_rows"].get("intent", 0))
    col_b.metric("Program Profile", system_summary["dataset_rows"].get("program_profiles", 0))
    col_c.metric("Roadmap", system_summary["dataset_rows"].get("roadmap", 0))

    st.markdown("#### Artifact")
    artifact_df = pd.DataFrame([
        {"komponen": "Intent model", "nilai": system_summary.get("intent_model", "-")},
        {"komponen": "Intent vectorizer", "nilai": system_summary.get("intent_vectorizer", "-")},
        {"komponen": "Recommender vectorizer", "nilai": system_summary.get("recommender_vectorizer", "-")},
        {"komponen": "Recommender matrix shape", "nilai": str(system_summary.get("recommender_matrix_shape", "-"))},
        {"komponen": "Metode recommender", "nilai": system_summary.get("recommender_method", "-")},
    ])
    st.dataframe(artifact_df, use_container_width=True, hide_index=True)

    st.markdown("#### Program Studi pada Knowledge Base")
    program_cols = [
        col for col in [
            "program_id", "nama_program_studi", "jenjang", "rumpun_ilmu",
            "bidang_keilmuan", "minat_cocok", "mapel_relevan"
        ] if col in bot.program_profiles_df.columns
    ]
    st.dataframe(bot.program_profiles_df[program_cols], use_container_width=True, hide_index=True)

    metrics_df = load_stage07_metrics(project_root)
    if not metrics_df.empty:
        st.markdown("#### Ringkasan Evaluasi Tahap 07")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("File evaluasi Tahap 07 belum ditemukan pada `reports/stage07/`. Bagian ini opsional.")

    with st.expander("Schema validation"):
        st.dataframe(bot.schema_check_df, use_container_width=True, hide_index=True)

with guide_tab:
    st.markdown("### Panduan Penggunaan")
    st.markdown(
        """
        **Cara menggunakan aplikasi:**

        1. Isi profil siswa pada tab Chatbot, atau langsung tulis pertanyaan di kolom chat.
        2. Masukkan informasi seperti mata pelajaran favorit, minat, hobi, gaya belajar, dan tujuan karier.
        3. Chatbot akan melakukan intent routing untuk menentukan apakah input meminta rekomendasi prodi, roadmap belajar, prospek karier, skill awal, atau informasi program studi.
        4. Untuk rekomendasi program studi, sistem menggunakan content-based filtering berbasis TF-IDF, cosine similarity, dan structured scoring.
        5. Hasil rekomendasi bersifat pendukung keputusan awal, bukan pengganti konsultasi akademik formal.
        """
    )

    st.markdown("### Batasan Akademik")
    st.info(
        "Dataset dan model pada project ini masih bersifat baseline akademik. "
        "Akurasi dan kualitas rekomendasi dapat meningkat jika dataset intent, profil program studi, roadmap, karier, dan skill diperluas serta divalidasi oleh domain expert."
    )

    st.markdown("### Struktur Input yang Disarankan")
    st.code(
        "Saya siswa SMA kelas 12. Saya suka matematika, statistik, dan komputer. "
        "Saya senang menganalisis data dan ingin menjadi Data Scientist. "
        "Program studi apa yang cocok untuk saya?",
        language="text",
    )
