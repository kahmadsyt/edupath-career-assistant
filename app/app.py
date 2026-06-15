# ============================================================
# EduPath Career Assistant
# Streamlit Chatbot Application
# ============================================================

import streamlit as st
from datetime import datetime

# Import fungsi chatbot dari file chatbot_utils.py
from chatbot_utils import chatbot_response


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="EduPath Career Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# DEFAULT STATE
# ============================================================

PROFILE_DEFAULTS = {
    "profile_jenjang": "",
    "profile_kelas": "",
    "profile_gaya_belajar": "",
    "profile_mata_pelajaran": "",
    "profile_minat": "",
    "profile_hobi": "",
    "profile_tujuan_karier": "",
}

APP_DEFAULTS = {
    "messages": [],
    "chat_history": [],
    "conversation_finished": False,
    "conversation_summary": "",
    "last_recommendation": None,
    "active_menu": "Chatbot",
}


def init_session_state():
    """
    Inisialisasi seluruh session_state agar aplikasi stabil
    ketika pertama kali dijalankan maupun setelah rerun.
    """
    for key, value in APP_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for key, value in PROFILE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all():
    """
    Reset seluruh percakapan dan seluruh input profil siswa.
    Fungsi ini memastikan form kembali kosong setelah tombol reset diklik.
    """
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.conversation_finished = False
    st.session_state.conversation_summary = ""
    st.session_state.last_recommendation = None
    st.session_state.active_menu = "Chatbot"

    for key, value in PROFILE_DEFAULTS.items():
        st.session_state[key] = value


def build_profile_prompt():
    """
    Mengubah input profil siswa menjadi prompt khusus rekomendasi.
    Marker [MODE:REKOMENDASI_PROFIL] dipakai agar chatbot_utils
    tidak salah membaca form profil sebagai intent roadmap/prospek.
    """
    jenjang = st.session_state.get("profile_jenjang", "")
    kelas = st.session_state.get("profile_kelas", "")
    gaya_belajar = st.session_state.get("profile_gaya_belajar", "")
    mata_pelajaran = st.session_state.get("profile_mata_pelajaran", "")
    minat = st.session_state.get("profile_minat", "")
    hobi = st.session_state.get("profile_hobi", "")
    tujuan_karier = st.session_state.get("profile_tujuan_karier", "")

    prompt = f"""
[MODE:REKOMENDASI_PROFIL]

Saya ingin mendapatkan rekomendasi program studi S1 berdasarkan profil siswa berikut:

Jenjang pendidikan: {jenjang}
Kelas: {kelas}
Gaya belajar dominan: {gaya_belajar}
Mata pelajaran favorit: {mata_pelajaran}
Minat utama: {minat}
Hobi atau aktivitas yang disukai: {hobi}
Tujuan karier: {tujuan_karier}

Tolong berikan rekomendasi program studi yang paling sesuai, alasan rekomendasi, prospek karier, skill awal, dan rencana belajar singkat.
""".strip()

    return prompt


def validate_profile_input():
    """
    Validasi sederhana agar rekomendasi profil tidak diproses
    ketika seluruh field masih kosong.
    """
    required_values = [
        st.session_state.get("profile_jenjang", ""),
        st.session_state.get("profile_kelas", ""),
        st.session_state.get("profile_gaya_belajar", ""),
        st.session_state.get("profile_mata_pelajaran", ""),
        st.session_state.get("profile_minat", ""),
        st.session_state.get("profile_hobi", ""),
        st.session_state.get("profile_tujuan_karier", ""),
    ]

    return any(str(value).strip() for value in required_values)


def normalize_chatbot_response(result):
    """
    Menormalkan output dari chatbot_utils.py agar tetap aman ditampilkan
    meskipun chatbot_response() mengembalikan string atau dictionary.

    Metadata teknis seperti intent, confidence, metode model, dan dataset
    sengaja tidak ditampilkan ke user.
    """
    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        response_text = (
            result.get("response")
            or result.get("answer")
            or result.get("message")
            or result.get("text")
            or ""
        )

        recommendations = result.get("recommendations") or result.get("rekomendasi")

        if recommendations and isinstance(recommendations, list):
            rec_text = "\n\n### Rekomendasi Program Studi\n"

            for idx, item in enumerate(recommendations, start=1):
                if isinstance(item, dict):
                    program = (
                        item.get("program_studi")
                        or item.get("nama_program_studi")
                        or item.get("prodi")
                        or item.get("name")
                        or "Program Studi"
                    )

                    alasan = item.get("alasan") or item.get("reason") or ""
                    prospek = item.get("prospek_karier") or item.get("career_prospect") or ""
                    roadmap = item.get("roadmap") or item.get("roadmap_belajar") or ""

                    rec_text += f"\n**{idx}. {program}**\n"

                    if alasan:
                        rec_text += f"\nAlasan: {alasan}\n"

                    if prospek:
                        rec_text += f"\nProspek karier: {prospek}\n"

                    if roadmap:
                        rec_text += f"\nRoadmap awal: {roadmap}\n"

                else:
                    rec_text += f"\n**{idx}. {item}**\n"

            response_text = f"{response_text}\n{rec_text}".strip()

        if not response_text:
            response_text = "Maaf, saya belum dapat menghasilkan respons yang sesuai."

        return response_text

    return str(result)


def get_bot_response(user_text):
    """
    Memanggil fungsi chatbot_response dari chatbot_utils.py.
    Dibuat fleksibel agar tetap berjalan meskipun fungsi hanya menerima satu parameter.
    """
    try:
        result = chatbot_response(user_text)
    except TypeError:
        result = chatbot_response(query=user_text)

    return normalize_chatbot_response(result)


def add_message(role, content):
    """
    Menambahkan pesan ke session_state.
    """
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M")
        }
    )


def process_user_message(user_text):
    """
    Memproses pesan user dan respons chatbot.
    """
    if not user_text or not user_text.strip():
        return

    st.session_state.conversation_finished = False
    st.session_state.conversation_summary = ""

    add_message("user", user_text)

    with st.spinner("EduPath sedang memproses jawaban..."):
        bot_reply = get_bot_response(user_text)

    add_message("assistant", bot_reply)
    st.session_state.last_recommendation = bot_reply


def build_conversation_summary():
    """
    Membuat ringkasan hasil percakapan secara sederhana dari chat history.
    Ringkasan ini hanya muncul setelah user menekan tombol Selesai Bertanya.
    """
    messages = st.session_state.get("messages", [])

    user_questions = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "user"
    ]

    assistant_answers = [
        msg["content"]
        for msg in messages
        if msg.get("role") == "assistant"
    ]

    profile_info = {
        "Jenjang": st.session_state.get("profile_jenjang", ""),
        "Kelas": st.session_state.get("profile_kelas", ""),
        "Gaya belajar": st.session_state.get("profile_gaya_belajar", ""),
        "Mata pelajaran favorit": st.session_state.get("profile_mata_pelajaran", ""),
        "Minat utama": st.session_state.get("profile_minat", ""),
        "Hobi": st.session_state.get("profile_hobi", ""),
        "Tujuan karier": st.session_state.get("profile_tujuan_karier", ""),
    }

    filled_profile = {
        key: value
        for key, value in profile_info.items()
        if str(value).strip()
    }

    summary = "# Ringkasan Hasil Percakapan EduPath Career Assistant\n\n"

    summary += "## Profil Siswa\n"
    if filled_profile:
        for key, value in filled_profile.items():
            summary += f"- **{key}:** {value}\n"
    else:
        summary += "- Profil siswa belum diisi melalui form.\n"

    summary += "\n## Pertanyaan yang Diajukan\n"
    if user_questions:
        for idx, question in enumerate(user_questions, start=1):
            summary += f"{idx}. {question}\n"
    else:
        summary += "- Belum ada pertanyaan yang diajukan.\n"

    summary += "\n## Hasil Respons Chatbot\n"
    if assistant_answers:
        latest_answer = assistant_answers[-1]
        summary += latest_answer
    else:
        summary += "Belum ada respons chatbot yang tersedia."

    summary += "\n\n## Catatan Penggunaan\n"
    summary += (
        "Hasil rekomendasi ini bersifat pendukung awal untuk membantu siswa "
        "memahami pilihan program studi, arah karier, dan roadmap belajar. "
        "Keputusan akhir tetap perlu mempertimbangkan minat mendalam, kemampuan akademik, "
        "diskusi dengan orang tua/guru BK, serta informasi resmi dari perguruan tinggi."
    )

    return summary


def finish_conversation():
    """
    Menandai bahwa user sudah selesai bertanya.
    Setelah ini menu Ringkasan Hasil Percakapan akan muncul.
    """
    st.session_state.conversation_finished = True
    st.session_state.conversation_summary = build_conversation_summary()
    st.session_state.active_menu = "Ringkasan Hasil Percakapan"


# ============================================================
# INISIALISASI
# ============================================================

init_session_state()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 EduPath")

menu_options = ["Chatbot", "Panduan"]

if st.session_state.get("conversation_finished"):
    menu_options.append("Ringkasan Hasil Percakapan")

if st.session_state.get("active_menu") not in menu_options:
    st.session_state.active_menu = "Chatbot"

selected_menu = st.sidebar.radio(
    "Menu",
    options=menu_options,
    key="active_menu"
)

st.sidebar.divider()

st.sidebar.button(
    "Reset Percakapan",
    on_click=reset_all,
    use_container_width=True
)


# ============================================================
# HALAMAN CHATBOT
# ============================================================

if selected_menu == "Chatbot":
    st.title("EduPath Career Assistant")
    st.caption(
        "Chatbot rekomendasi program studi S1 dan roadmap belajar untuk siswa SMA/SMK/MA."
    )

    st.divider()

    st.subheader("Input Profil Siswa SMA/SMK/MA")
    st.write(
        "Isi profil singkat berikut untuk memperoleh rekomendasi program studi yang lebih terarah."
    )

    with st.form("profile_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.selectbox(
                "Jenjang",
                options=["", "SMA", "SMK", "MA"],
                key="profile_jenjang",
                format_func=lambda x: "Pilih jenjang" if x == "" else x
            )

        with col2:
            st.selectbox(
                "Kelas",
                options=["", "10", "11", "12"],
                key="profile_kelas",
                format_func=lambda x: "Pilih kelas" if x == "" else x
            )

        with col3:
            st.selectbox(
                "Gaya belajar dominan",
                options=[
                    "",
                    "logis dan analitis",
                    "visual",
                    "praktik langsung",
                    "membaca dan menulis",
                    "diskusi dan kolaboratif"
                ],
                key="profile_gaya_belajar",
                format_func=lambda x: "Pilih gaya belajar" if x == "" else x
            )

        st.text_input(
            "Mata pelajaran favorit",
            key="profile_mata_pelajaran",
            placeholder="Contoh: Matematika, IPA, Bahasa Inggris"
        )

        st.text_input(
            "Minat utama",
            key="profile_minat",
            placeholder="Contoh: teknologi, desain, bisnis, kesehatan"
        )

        st.text_input(
            "Hobi atau aktivitas yang disukai",
            key="profile_hobi",
            placeholder="Contoh: menggambar, coding, membaca, membuat konten"
        )

        st.text_input(
            "Tujuan karier",
            key="profile_tujuan_karier",
            placeholder="Contoh: Data Analyst, Software Engineer, Designer"
        )

        submitted_profile = st.form_submit_button(
            "Dapatkan Rekomendasi dari Profil",
            use_container_width=True
        )

    if submitted_profile:
        if validate_profile_input():
            profile_prompt = build_profile_prompt()
            process_user_message(profile_prompt)
            st.rerun()
        else:
            st.warning("Silakan isi minimal satu informasi profil terlebih dahulu.")

    st.divider()

    st.subheader("Percakapan")

    if not st.session_state.messages:
        st.info(
            "Silakan isi profil siswa atau ajukan pertanyaan langsung, misalnya: "
            "Saya suka matematika dan komputer, cocoknya ambil jurusan apa?"
        )

    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input(
        "Tulis pertanyaan tentang program studi, minat, karier, atau roadmap belajar..."
    )

    if user_input:
        process_user_message(user_input)
        st.rerun()

    if st.session_state.messages:
        st.divider()

        col_finish, col_reset = st.columns([1, 1])

        with col_finish:
            st.button(
                "Selesai Bertanya",
                on_click=finish_conversation,
                use_container_width=True
            )

        with col_reset:
            st.button(
                "Reset Percakapan dan Form",
                on_click=reset_all,
                use_container_width=True
            )


# ============================================================
# HALAMAN PANDUAN
# ============================================================

elif selected_menu == "Panduan":
    st.title("Panduan Penggunaan")
    st.write(
        "Halaman ini berisi panduan singkat penggunaan EduPath Career Assistant."
    )

    st.markdown(
        """
### 1. Gunakan Form Profil

Isi data seperti jenjang, kelas, gaya belajar, mata pelajaran favorit, minat, hobi, dan tujuan karier.  
Setelah itu klik **Dapatkan Rekomendasi dari Profil**.

### 2. Gunakan Pertanyaan Bebas

Selain form, pengguna juga dapat langsung bertanya melalui kolom chat.

Contoh pertanyaan:

- Saya suka matematika dan komputer, cocoknya ambil jurusan apa?
- Saya anak SMK RPL, ingin jadi Data Analyst, sebaiknya kuliah apa?
- Saya suka desain dan menggambar, program studi apa yang cocok?
- Saya ingin bekerja di bidang teknologi, roadmap belajarnya bagaimana?

### 3. Lihat Ringkasan Hasil Percakapan

Setelah selesai bertanya, klik tombol **Selesai Bertanya**.  
Menu **Ringkasan Hasil Percakapan** akan muncul secara otomatis di sidebar.

### 4. Reset Percakapan

Klik **Reset Percakapan** untuk menghapus seluruh histori chat dan mengosongkan kembali form profil siswa.
        """
    )


# ============================================================
# HALAMAN RINGKASAN HASIL PERCAKAPAN
# ============================================================

elif selected_menu == "Ringkasan Hasil Percakapan":
    st.title("Ringkasan Hasil Percakapan")

    if st.session_state.get("conversation_finished") and st.session_state.get("conversation_summary"):
        st.markdown(st.session_state.conversation_summary)

        st.download_button(
            label="Download Ringkasan Markdown",
            data=st.session_state.conversation_summary,
            file_name="ringkasan_hasil_percakapan_edupath.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info(
            "Ringkasan belum tersedia. Silakan kembali ke menu Chatbot dan klik tombol Selesai Bertanya."
        )