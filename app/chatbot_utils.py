"""
Utility module for EduPath Career Assistant Streamlit application.

This module integrates:
1. Intent classification model from Stage 04.
2. Content-based program recommender from Stage 05.
3. Chatbot pipeline integration from Stage 06.
4. Error-aware response handling for Stage 08 Streamlit application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

STOPWORDS = {
    "saya", "aku", "anda", "kamu", "dia", "mereka", "kami", "kita",
    "yang", "dan", "atau", "di", "ke", "dari", "untuk", "dengan", "pada",
    "adalah", "itu", "ini", "apa", "bagaimana", "kalau", "jika", "agar",
    "mau", "ingin", "bisa", "dapat", "harus", "dulu", "ya", "dong",
    "nih", "sih", "lah", "pun", "nya", "sebagai", "dalam", "bidang",
    "masuk", "ambil", "pilih", "cocoknya", "cocok", "pas"
}

EXTRA_NORMALIZATION = {
    "gw": "saya",
    "gue": "saya",
    "gua": "saya",
    "aku": "saya",
    "ngoding": "pemrograman",
    "coding": "pemrograman",
    "ngode": "pemrograman",
    "prodi": "program studi",
    "jurusan": "program studi",
    "kuliah": "program studi",
    "programmer": "software engineer",
    "developer": "software engineer",
    "it": "informatika",
    "ti": "teknik informatika",
    "dkv": "desain komunikasi visual",
    "uiux": "ui ux",
    "ui": "user interface",
    "ux": "user experience",
    "seneng": "suka",
    "suka": "suka",
    "resep": "suka",
    "dadi": "menjadi",
    "janten": "menjadi",
    "pengin": "ingin",
    "pengen": "ingin",
    "hoyong": "ingin",
    "naon": "apa",
    "opo": "apa",
    "kumaha": "bagaimana",
    "gimana": "bagaimana",
}

INTENT_RULES = {
    "sapaan": [
        "halo", "hai", "hello", "assalamualaikum", "selamat pagi",
        "selamat siang", "selamat sore", "selamat malam"
    ],
    "roadmap_belajar": [
        "roadmap", "alur belajar", "belajar apa", "harus belajar",
        "mulai dari mana", "persiapan", "materi awal", "tahapan belajar"
    ],
    "prospek_karier": [
        "prospek kerja", "karier", "pekerjaan", "kerja apa",
        "peluang kerja", "profesi", "job", "masa depan"
    ],
    "skill_awal": [
        "skill", "kemampuan", "kompetensi", "belajar dulu",
        "materi dasar", "dasar apa", "tools"
    ],
    "info_program_studi": [
        "apa itu", "jelaskan", "informasi", "info",
        "mempelajari apa", "belajar apa di"
    ],
    "rekomendasi_prodi": [
        "rekomendasi", "program studi", "prodi", "jurusan",
        "cocok", "pilih", "kuliah apa", "suka", "minat"
    ],
    "klarifikasi_minat": [
        "bingung", "belum tahu", "belum tau", "tidak tahu",
        "ga tau", "gak tau", "masih ragu"
    ],
}

ROUTING_PRIORITY = [
    "sapaan",
    "roadmap_belajar",
    "prospek_karier",
    "skill_awal",
    "info_program_studi",
    "rekomendasi_prodi",
    "klarifikasi_minat",
]

REQUIRED_DATASET_COLUMNS = {
    "intent": ["utterance", "intent_label", "utterance_preprocessed"],
    "program_profiles": ["program_id", "nama_program_studi", "deskripsi_singkat", "program_profile_text"],
    "career": ["karier_id", "nama_karier", "program_studi_relevan_id_list"],
    "skill": ["skill_id", "nama_skill", "program_studi_relevan_id_list"],
    "roadmap": ["program_id", "fase", "urutan_fase", "tujuan_fase", "materi_pokok"],
    "normalisasi": ["kata_input_clean", "kata_baku_clean"],
}

PROFILE_FORM_MARKERS = [
    "jenjang pendidikan:",
    "kelas:",
    "gaya belajar dominan:",
    "mata pelajaran favorit:",
    "minat utama:",
    "hobi atau aktivitas yang disukai:",
    "tujuan karier:",
]

DOMAIN_FALLBACKS = {
    "kesehatan": {
        "keywords": [
            "kesehatan", "dokter", "kedokteran", "medis", "perawat",
            "keperawatan", "farmasi", "gizi", "rumah sakit", "biologi"
        ],
        "programs": [
            {
                "nama_program_studi": "Kedokteran",
                "bidang": "Kesehatan dan ilmu medis",
                "alasan": "paling relevan untuk tujuan karier menjadi dokter.",
                "prospek": "Dokter umum, dokter spesialis setelah pendidikan lanjutan, peneliti kesehatan, tenaga medis klinis.",
                "skill": "Biologi dasar, kimia dasar, anatomi dasar, komunikasi pasien, etika profesi."
            },
            {
                "nama_program_studi": "Keperawatan",
                "bidang": "Kesehatan dan pelayanan pasien",
                "alasan": "cocok untuk siswa yang tertarik pada pelayanan kesehatan dan pendampingan pasien.",
                "prospek": "Perawat, perawat klinis, perawat komunitas, edukator kesehatan.",
                "skill": "Dasar keperawatan, komunikasi empatik, keselamatan pasien, observasi klinis."
            },
            {
                "nama_program_studi": "Kesehatan Masyarakat",
                "bidang": "Kesehatan publik",
                "alasan": "cocok jika minatnya pada pencegahan penyakit, edukasi kesehatan, dan analisis masalah kesehatan masyarakat.",
                "prospek": "Analis kesehatan masyarakat, epidemiology assistant, promotor kesehatan, health program officer.",
                "skill": "Statistik dasar, epidemiologi dasar, promosi kesehatan, analisis data kesehatan."
            },
            {
                "nama_program_studi": "Farmasi",
                "bidang": "Obat dan pelayanan kefarmasian",
                "alasan": "relevan untuk siswa IPA yang tertarik pada obat, kimia, dan layanan kesehatan.",
                "prospek": "Apoteker setelah pendidikan profesi, staf farmasi, quality control farmasi, medical representative.",
                "skill": "Kimia dasar, biologi dasar, farmakologi dasar, ketelitian laboratorium."
            },
            {
                "nama_program_studi": "Gizi",
                "bidang": "Kesehatan dan nutrisi",
                "alasan": "sesuai untuk minat kesehatan yang berfokus pada pola makan, nutrisi, dan pencegahan penyakit.",
                "prospek": "Nutritionist, konsultan gizi, staf gizi rumah sakit, edukator nutrisi.",
                "skill": "Biologi dasar, ilmu gizi dasar, komunikasi edukatif, analisis pola makan."
            },
        ],
    }
}


@dataclass
class ProjectPaths:
    """Resolved project paths used by the Streamlit application."""

    project_root: Path
    data_dir: Path
    model_dir: Path
    report_dir: Path


def resolve_project_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """Resolve project, data, model, and report paths.

    The application is normally executed using:
    streamlit run app/app.py

    In that case, this file is located at <project_root>/app/chatbot_utils.py,
    therefore project_root defaults to parent of the app directory.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root).resolve()

    data_dir = project_root / "data" / "processed"

    # Fallback for exported ZIP files that use data-processed.
    if not data_dir.exists() and (project_root / "data-processed").exists():
        data_dir = project_root / "data-processed"

    model_dir = project_root / "models"
    report_dir = project_root / "reports"

    return ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        model_dir=model_dir,
        report_dir=report_dir,
    )


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_files(paths: Dict[str, Path], label: str) -> None:
    """Validate required files and raise clear error if missing."""
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"File {label} berikut belum ditemukan:\n" + "\n".join(missing)
        )


def validate_dataset_schema(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Validate minimum schema required by the chatbot pipeline."""
    rows = []

    for dataset_name, columns in REQUIRED_DATASET_COLUMNS.items():
        df = datasets[dataset_name]
        for column in columns:
            rows.append({
                "dataset": dataset_name,
                "required_column": column,
                "status": "OK" if column in df.columns else "MISSING",
            })

    schema_df = pd.DataFrame(rows)

    if (schema_df["status"] == "MISSING").any():
        missing_rows = schema_df[schema_df["status"] == "MISSING"]
        raise ValueError(
            "Terdapat kolom wajib yang belum tersedia:\n"
            + missing_rows.to_string(index=False)
        )

    return schema_df


def safe_text(value: Any, default: str = "-") -> str:
    """Convert value to clean string for display."""
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def split_values(value: Any) -> List[str]:
    """Split string values separated by pipe or comma into list."""
    if value is None or pd.isna(value):
        return []
    value = str(value).strip()
    if not value:
        return []
    return [item.strip() for item in re.split(r"\||,", value) if item.strip()]


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert decision scores into pseudo probability."""
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denominator = exp_values.sum()
    if denominator == 0:
        return np.ones_like(exp_values) / len(exp_values)
    return exp_values / denominator


class EduPathChatbot:
    """End-to-end chatbot pipeline for EduPath Career Assistant."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.paths = resolve_project_paths(project_root)
        self.dataset_paths = {
            "intent": self.paths.data_dir / "intent_dataset_processed.csv",
            "program": self.paths.data_dir / "program_studi_s1_processed.csv",
            "program_profiles": self.paths.data_dir / "program_recommender_profiles_stage05.csv",
            "career": self.paths.data_dir / "karier_processed.csv",
            "skill": self.paths.data_dir / "skill_awal_processed.csv",
            "roadmap": self.paths.data_dir / "roadmap_belajar_processed.csv",
            "normalisasi": self.paths.data_dir / "normalisasi_bahasa_processed.csv",
        }
        self.artifact_paths = {
            "intent_model": self.paths.model_dir / "intent_classifier_best_model.joblib",
            "intent_vectorizer": self.paths.model_dir / "intent_tfidf_vectorizer.joblib",
            "intent_metadata": self.paths.model_dir / "intent_classifier_metadata.json",
            "recommender_vectorizer": self.paths.model_dir / "program_recommender_tfidf_vectorizer.joblib",
            "recommender_matrix": self.paths.model_dir / "program_recommender_matrix.joblib",
            "recommender_config": self.paths.model_dir / "program_recommender_config_stage05.json",
        }

        self._load_datasets()
        self._load_artifacts()
        self.schema_check_df = validate_dataset_schema(self.datasets)
        self.normalization_map = self._build_normalization_map()

    # ============================================================
    # Loading
    # ============================================================

    def _load_datasets(self) -> None:
        validate_files(self.dataset_paths, "dataset processed")

        self.intent_df = pd.read_csv(self.dataset_paths["intent"])
        self.program_df = pd.read_csv(self.dataset_paths["program"])
        self.program_profiles_df = pd.read_csv(self.dataset_paths["program_profiles"])
        self.career_df = pd.read_csv(self.dataset_paths["career"])
        self.skill_df = pd.read_csv(self.dataset_paths["skill"])
        self.roadmap_df = pd.read_csv(self.dataset_paths["roadmap"])
        self.normalisasi_df = pd.read_csv(self.dataset_paths["normalisasi"])

        self.datasets = {
            "intent": self.intent_df,
            "program": self.program_df,
            "program_profiles": self.program_profiles_df,
            "career": self.career_df,
            "skill": self.skill_df,
            "roadmap": self.roadmap_df,
            "normalisasi": self.normalisasi_df,
        }

    def _load_artifacts(self) -> None:
        validate_files(self.artifact_paths, "model artifact")

        self.intent_model = joblib.load(self.artifact_paths["intent_model"])
        self.intent_vectorizer = joblib.load(self.artifact_paths["intent_vectorizer"])
        self.intent_metadata = read_json(self.artifact_paths["intent_metadata"])
        self.recommender_vectorizer = joblib.load(self.artifact_paths["recommender_vectorizer"])
        self.recommender_matrix = joblib.load(self.artifact_paths["recommender_matrix"])
        self.recommender_config = read_json(self.artifact_paths["recommender_config"])

        matrix_rows = getattr(self.recommender_matrix, "shape", [0])[0]
        if matrix_rows != len(self.program_profiles_df):
            raise ValueError(
                "Jumlah baris matrix recommender tidak sama dengan jumlah profil program. "
                f"Matrix rows={matrix_rows}, profile rows={len(self.program_profiles_df)}."
            )

        self.text_similarity_weight = float(
            self.recommender_config.get("text_similarity_weight", 0.65)
        )
        self.structured_score_weight = float(
            self.recommender_config.get("structured_score_weight", 0.35)
        )

    # ============================================================
    # Preprocessing
    # ============================================================

    def _build_normalization_map(self) -> Dict[str, str]:
        norm_map: Dict[str, str] = {}

        for _, row in self.normalisasi_df.iterrows():
            key = safe_text(row.get("kata_input_clean"), "").lower()
            value = safe_text(row.get("kata_baku_clean"), "").lower()
            if key and value and key != "nan" and value != "nan":
                norm_map[key] = value

        norm_map.update(EXTRA_NORMALIZATION)
        return norm_map

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_text(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = cleaned.split()

        normalized_tokens: List[str] = []
        for token in tokens:
            replacement = self.normalization_map.get(token, token)
            normalized_tokens.extend(str(replacement).split())

        return " ".join(normalized_tokens)

    def preprocess_text(self, text: str) -> str:
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        filtered_tokens = [
            token for token in tokens
            if token not in STOPWORDS and len(token) > 1
        ]
        return " ".join(filtered_tokens)

    # ============================================================
    # Intent Classification and Routing
    # ============================================================

    def predict_intent(self, user_text: str) -> Dict[str, Any]:
        processed_text = self.preprocess_text(user_text)
        vectorized_text = self.intent_vectorizer.transform([processed_text])

        predicted_label = str(self.intent_model.predict(vectorized_text)[0])
        labels = list(
            getattr(self.intent_model, "classes_", self.intent_metadata.get("labels", []))
        )

        if hasattr(self.intent_model, "predict_proba"):
            probabilities = self.intent_model.predict_proba(vectorized_text)[0]
            confidence = float(np.max(probabilities))
            top_intents = sorted(
                [
                    {"intent": str(label), "score": float(score)}
                    for label, score in zip(labels, probabilities)
                ],
                key=lambda item: item["score"],
                reverse=True,
            )
        elif hasattr(self.intent_model, "decision_function"):
            decision_scores = self.intent_model.decision_function(vectorized_text)
            if np.asarray(decision_scores).ndim == 1:
                decision_scores = np.asarray(decision_scores).reshape(1, -1)
            pseudo_probs = softmax(decision_scores[0])
            confidence = float(np.max(pseudo_probs))
            top_intents = sorted(
                [
                    {"intent": str(label), "score": float(score)}
                    for label, score in zip(labels, pseudo_probs)
                ],
                key=lambda item: item["score"],
                reverse=True,
            )
        else:
            confidence = 1.0
            top_intents = [{"intent": predicted_label, "score": 1.0}]

        return {
            "raw_text": user_text,
            "processed_text": processed_text,
            "predicted_intent": predicted_label,
            "confidence": round(confidence, 4),
            "top_intents": top_intents[:5],
        }

    def rule_based_intent(self, user_text: str) -> Dict[str, Any]:
        raw = self.clean_text(user_text)
        normalized = self.normalize_text(user_text)
        processed = self.preprocess_text(user_text)
        combined_text = f"{raw} {normalized} {processed}"

        for intent in ROUTING_PRIORITY:
            matched_keywords = []
            for keyword in INTENT_RULES[intent]:
                keyword_clean = self.clean_text(keyword)
                if keyword_clean in combined_text:
                    matched_keywords.append(keyword)

            if matched_keywords:
                score = min(1.0, 0.4 + 0.2 * len(matched_keywords))
                return {
                    "rule_intent": intent,
                    "rule_confidence": round(score, 4),
                    "matched_keywords": matched_keywords,
                }

        return {
            "rule_intent": None,
            "rule_confidence": 0.0,
            "matched_keywords": [],
        }

    def is_profile_form_query(self, user_text: str) -> bool:
        """Detect profile-form prompts sent from Streamlit.

        The form prompt often contains words such as "roadmap" or "karier".
        Without this guard, rule-based routing can incorrectly classify a
        profile recommendation request as roadmap_belajar or prospek_karier.
        """
        text = str(user_text).lower()

        if "[mode:rekomendasi_profil]" in text:
            return True

        marker_count = sum(1 for marker in PROFILE_FORM_MARKERS if marker in text)
        has_recommendation_phrase = (
            "rekomendasi program studi" in text
            or "rekomendasi prodi" in text
            or "berdasarkan profil" in text
        )

        return marker_count >= 3 and has_recommendation_phrase

    def route_intent(self, user_text: str, model_threshold: float = 0.30) -> Dict[str, Any]:
        model_result = self.predict_intent(user_text)
        rule_result = self.rule_based_intent(user_text)

        if self.is_profile_form_query(user_text):
            return {
                "final_intent": "rekomendasi_prodi",
                "routing_source": "profile_form",
                "model_result": model_result,
                "rule_result": rule_result,
            }

        model_intent = model_result["predicted_intent"]
        model_confidence = float(model_result["confidence"])
        rule_intent = rule_result["rule_intent"]
        rule_confidence = float(rule_result["rule_confidence"])

        if rule_intent is not None and rule_confidence >= 0.40:
            final_intent = rule_intent
            source = "rule_based"
        elif model_confidence >= model_threshold:
            final_intent = model_intent
            source = "model"
        else:
            final_intent = "fallback"
            source = "fallback_threshold"

        return {
            "final_intent": final_intent,
            "routing_source": source,
            "model_result": model_result,
            "rule_result": rule_result,
        }

    # ============================================================
    # Mapping and Recommender
    # ============================================================

    def get_career_names_from_program(self, program_row: pd.Series) -> List[str]:
        if "mapped_career_names" in program_row.index and pd.notna(program_row["mapped_career_names"]):
            return split_values(str(program_row["mapped_career_names"]).replace(", ", "|"))

        career_ids = split_values(program_row.get("prospek_karier_id_list", ""))
        names = self.career_df[self.career_df["karier_id"].isin(career_ids)]["nama_karier"].tolist()
        return names

    def get_skill_names_from_program(self, program_row: pd.Series) -> List[str]:
        if "mapped_skill_names" in program_row.index and pd.notna(program_row["mapped_skill_names"]):
            return split_values(str(program_row["mapped_skill_names"]).replace(", ", "|"))

        skill_ids = split_values(program_row.get("skill_id_list", ""))
        names = self.skill_df[self.skill_df["skill_id"].isin(skill_ids)]["nama_skill"].tolist()
        return names

    def get_roadmap_by_program_id(self, program_id: str) -> pd.DataFrame:
        roadmap = self.roadmap_df[self.roadmap_df["program_id"] == program_id].copy()
        if "urutan_fase" in roadmap.columns:
            roadmap = roadmap.sort_values("urutan_fase")
        return roadmap.reset_index(drop=True)

    def build_structured_profile_text(self, row: pd.Series) -> str:
        candidate_columns = [
            "nama_program_studi",
            "rumpun_ilmu",
            "bidang_keilmuan",
            "deskripsi_singkat",
            "minat_cocok",
            "mapel_relevan",
            "hobi_relevan",
            "gaya_belajar_cocok",
            "karakteristik_cocok",
            "keyword_rekomendasi",
            "mapped_career_text",
            "mapped_career_names",
            "mapped_skill_names",
        ]

        values = []
        for column in candidate_columns:
            if column in row.index and pd.notna(row[column]):
                values.append(str(row[column]))

        return " ".join(values)

    def keyword_overlap(self, query_text: str, profile_text: str) -> Tuple[float, List[str]]:
        query_tokens = set(self.preprocess_text(query_text).split())
        profile_tokens = set(self.preprocess_text(profile_text).split())

        if not query_tokens:
            return 0.0, []

        matched_tokens = sorted(list(query_tokens.intersection(profile_tokens)))
        score = len(matched_tokens) / max(1, min(len(query_tokens), 10))
        return min(score, 1.0), matched_tokens

    def recommend_programs(self, user_text: str, top_n: int = 3) -> pd.DataFrame:
        processed_query = self.preprocess_text(user_text)
        if not processed_query:
            processed_query = self.normalize_text(user_text)

        query_vector = self.recommender_vectorizer.transform([processed_query])
        text_similarities = cosine_similarity(query_vector, self.recommender_matrix).ravel()

        recommendations = []
        for idx, row in self.program_profiles_df.iterrows():
            structured_text = self.build_structured_profile_text(row)
            structured_score, matched_tokens = self.keyword_overlap(user_text, structured_text)
            text_similarity = float(text_similarities[idx])
            final_score = (
                self.text_similarity_weight * text_similarity
                + self.structured_score_weight * structured_score
            )

            career_names = self.get_career_names_from_program(row)
            skill_names = self.get_skill_names_from_program(row)

            recommendations.append({
                "rank": None,
                "program_id": row.get("program_id"),
                "nama_program_studi": row.get("nama_program_studi"),
                "jenjang": row.get("jenjang"),
                "rumpun_ilmu": row.get("rumpun_ilmu"),
                "bidang_keilmuan": row.get("bidang_keilmuan"),
                "deskripsi_singkat": row.get("deskripsi_singkat"),
                "text_similarity": round(text_similarity, 4),
                "structured_score": round(structured_score, 4),
                "final_score": round(final_score, 4),
                "matched_keywords": ", ".join(matched_tokens[:12]),
                "prospek_karier": ", ".join(career_names),
                "skill_awal": ", ".join(skill_names),
            })

        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
        rec_df["rank"] = rec_df.index + 1
        return rec_df

    def find_program_from_text(self, user_text: str) -> Optional[pd.Series]:
        raw_clean = self.clean_text(user_text)
        processed = self.preprocess_text(user_text)

        for _, row in self.program_profiles_df.iterrows():
            program_name = self.clean_text(row.get("nama_program_studi", ""))
            if program_name and program_name in raw_clean:
                return row

        best_row = None
        best_score = 0.0

        for _, row in self.program_profiles_df.iterrows():
            keyword_text = " ".join([
                safe_text(row.get("nama_program_studi"), ""),
                safe_text(row.get("keyword_rekomendasi"), ""),
                safe_text(row.get("bidang_keilmuan"), ""),
                safe_text(row.get("program_profile_text"), ""),
            ])
            score, _ = self.keyword_overlap(processed, keyword_text)
            if score > best_score:
                best_score = score
                best_row = row

        if best_score >= 0.20:
            return best_row

        top_rec = self.recommend_programs(user_text, top_n=1)
        if not top_rec.empty and float(top_rec.iloc[0]["final_score"]) > 0:
            program_id = top_rec.iloc[0]["program_id"]
            matched = self.program_profiles_df[self.program_profiles_df["program_id"] == program_id]
            if not matched.empty:
                return matched.iloc[0]

        return None

    # ============================================================
    # Domain Coverage Guard
    # ============================================================

    def dataset_contains_domain_keywords(self, keywords: List[str]) -> bool:
        """Check whether program profile dataset contains a requested domain."""
        profile_text = " ".join(
            self.program_profiles_df.astype(str).fillna("").agg(" ".join, axis=1).tolist()
        ).lower()
        return any(keyword.lower() in profile_text for keyword in keywords)

    def detect_domain_gap(self, user_text: str) -> Optional[str]:
        """Detect a domain requested by user but not covered by current dataset."""
        normalized = self.normalize_text(user_text).lower()
        raw = self.clean_text(user_text).lower()
        combined = f"{raw} {normalized}"

        for domain, config in DOMAIN_FALLBACKS.items():
            keywords = config.get("keywords", [])
            user_mentions_domain = any(keyword.lower() in combined for keyword in keywords)
            dataset_has_domain = self.dataset_contains_domain_keywords(keywords)

            if user_mentions_domain and not dataset_has_domain:
                return domain

        return None

    @staticmethod
    def format_generic_roadmap(program_name: str) -> List[str]:
        """Create a simple generic roadmap when detailed roadmap data is unavailable."""
        return [
            f"1. Pahami gambaran umum {program_name}, mata kuliah inti, dan prospek kariernya.",
            "2. Perkuat mata pelajaran pendukung di sekolah, terutama yang relevan dengan bidang tersebut.",
            "3. Ikuti aktivitas praktik sederhana, membaca sumber pengantar, dan mulai membuat catatan/portofolio belajar.",
            "4. Validasi pilihan melalui guru BK, orang tua, alumni, atau informasi resmi kampus.",
        ]

    def format_domain_gap_response(self, domain: str) -> Tuple[str, pd.DataFrame]:
        """Return a useful answer when the requested domain is outside dataset coverage."""
        config = DOMAIN_FALLBACKS[domain]
        programs = config["programs"]
        fallback_df = pd.DataFrame(programs)
        fallback_df.insert(0, "rank", range(1, len(fallback_df) + 1))

        lines = [
            "Berdasarkan profil yang disampaikan, minat Anda lebih mengarah ke rumpun kesehatan.",
            "Namun, dataset rekomendasi yang sedang digunakan aplikasi saat ini belum memiliki data program studi kesehatan yang cukup lengkap. Karena itu, berikut rekomendasi akademik umum yang lebih sesuai:\n",
        ]

        for idx, item in enumerate(programs, start=1):
            lines.append(f"{idx}. {item['nama_program_studi']}")
            lines.append(f"   - Bidang: {item['bidang']}")
            lines.append(f"   - Alasan: {item['alasan']}")
            lines.append(f"   - Prospek karier: {item['prospek']}")
            lines.append(f"   - Skill awal: {item['skill']}")
            lines.append("")

        lines.append("Roadmap belajar awal yang disarankan:")
        lines.append("1. Perkuat Biologi, Kimia, Matematika dasar, dan Bahasa Inggris.")
        lines.append("2. Mulai membaca pengantar anatomi, kesehatan manusia, etika profesi, dan literasi kesehatan.")
        lines.append("3. Ikuti kegiatan pendukung seperti PMR, karya ilmiah remaja, seminar kesehatan, atau observasi profesi kesehatan.")
        lines.append("4. Cek syarat masuk resmi dari kampus tujuan karena beberapa program kesehatan memiliki seleksi dan ketentuan khusus.")
        lines.append("")
        lines.append(
            "Catatan: agar aplikasi dapat memberi rekomendasi berbasis dataset untuk kasus seperti ini, "
            "dataset program studi sebaiknya ditambah dengan rumpun Kedokteran, Keperawatan, Farmasi, Gizi, dan Kesehatan Masyarakat."
        )

        return "\n".join(lines), fallback_df

    def build_natural_reason(self, row: pd.Series) -> str:
        """Build user-facing recommendation reason without technical scores."""
        parts = []

        bidang = safe_text(row.get("bidang_keilmuan"), "")
        minat = safe_text(row.get("minat_cocok"), "")
        mapel = safe_text(row.get("mapel_relevan"), "")
        hobi = safe_text(row.get("hobi_relevan"), "")
        gaya = safe_text(row.get("gaya_belajar_cocok"), "")

        if bidang and bidang != "-":
            parts.append(f"berada pada bidang {bidang}")
        if minat and minat != "-":
            parts.append(f"relevan dengan minat seperti {minat}")
        if mapel and mapel != "-":
            parts.append(f"didukung oleh mata pelajaran {mapel}")
        if hobi and hobi != "-":
            parts.append(f"selaras dengan aktivitas seperti {hobi}")
        if gaya and gaya != "-":
            parts.append(f"cocok untuk gaya belajar {gaya}")

        if not parts:
            return "program ini memiliki kedekatan dengan profil minat dan tujuan karier yang Anda sampaikan."

        return "; ".join(parts[:3]) + "."

    def append_top_program_roadmap(self, lines: List[str], rec_df: pd.DataFrame) -> None:
        """Append roadmap for the first recommendation without failing when roadmap is unavailable."""
        if rec_df.empty:
            return

        top_program_id = rec_df.iloc[0].get("program_id")
        top_program_name = safe_text(rec_df.iloc[0].get("nama_program_studi"), "program studi pilihan utama")
        roadmap = self.get_roadmap_by_program_id(str(top_program_id))

        lines.append("Roadmap belajar awal untuk rekomendasi prioritas utama:")

        if roadmap.empty:
            for item in self.format_generic_roadmap(top_program_name):
                lines.append(f"- {item}")
            lines.append("")
            return

        for _, row in roadmap.head(3).iterrows():
            fase = safe_text(row.get("fase"))
            tujuan = safe_text(row.get("tujuan_fase"))
            materi = safe_text(row.get("materi_pokok"))
            lines.append(f"- {fase}: {tujuan} Materi awal: {materi}.")
        lines.append("")

    # ============================================================
    # Response Formatting
    # ============================================================

    def format_recommendation_response(self, user_text: str, top_n: int = 3) -> Tuple[str, pd.DataFrame]:
        domain_gap = self.detect_domain_gap(user_text)
        if domain_gap is not None:
            return self.format_domain_gap_response(domain_gap)

        rec_df = self.recommend_programs(user_text, top_n=top_n)

        if rec_df.empty:
            return (
                "Saya belum dapat memberikan rekomendasi program studi. "
                "Coba jelaskan minat, mata pelajaran favorit, hobi, gaya belajar, atau tujuan karier Anda.",
                rec_df,
            )

        lines = ["Berikut rekomendasi awal program studi berdasarkan profil yang Anda sampaikan:\n"]

        for _, row in rec_df.iterrows():
            lines.append(f"{int(row['rank'])}. {row['nama_program_studi']} ({safe_text(row.get('jenjang'))})")
            lines.append(f"   - Bidang: {safe_text(row.get('bidang_keilmuan'))}")
            lines.append(f"   - Alasan: {self.build_natural_reason(row)}")
            lines.append(f"   - Prospek karier: {safe_text(row.get('prospek_karier'))}")
            lines.append(f"   - Skill awal: {safe_text(row.get('skill_awal'))}")
            lines.append("")

        self.append_top_program_roadmap(lines, rec_df)

        lines.append(
            "Catatan: rekomendasi ini bersifat awal dan sebaiknya divalidasi kembali "
            "dengan minat, nilai akademik, portofolio, serta konsultasi dengan guru BK atau pembimbing akademik."
        )

        return "\n".join(lines), rec_df

    def format_roadmap_response(self, user_text: str) -> Tuple[str, pd.DataFrame]:
        domain_gap = self.detect_domain_gap(user_text)
        if domain_gap is not None:
            response_text, payload = self.format_domain_gap_response(domain_gap)
            return response_text, payload

        program = self.find_program_from_text(user_text)

        if program is None:
            return (
                "Saya belum dapat menentukan program studi yang dimaksud. "
                "Mohon sebutkan program studi atau minat Anda, misalnya Sains Data, Teknik Informatika, atau Sistem Informasi.",
                pd.DataFrame(),
            )

        roadmap = self.get_roadmap_by_program_id(program["program_id"])
        if roadmap.empty:
            lines = [
                f"Roadmap khusus untuk {program['nama_program_studi']} belum tersedia pada dataset saat ini. "
                "Berikut roadmap umum yang tetap dapat digunakan:\n"
            ]
            for item in self.format_generic_roadmap(str(program["nama_program_studi"])):
                lines.append(f"- {item}")
            return "\n".join(lines), roadmap

        lines = [f"Roadmap belajar awal untuk {program['nama_program_studi']} ({program['jenjang']}):\n"]

        for _, row in roadmap.iterrows():
            lines.append(f"Fase {row['urutan_fase']} — {row['fase']} ({row['durasi_rekomendasi']})")
            lines.append(f"- Tujuan: {row['tujuan_fase']}")
            lines.append(f"- Materi pokok: {row['materi_pokok']}")
            lines.append(f"- Aktivitas praktik: {row['aktivitas_praktik']}")
            lines.append(f"- Output portofolio: {row['output_portofolio']}")
            lines.append(f"- Tools: {row['tools_umum']}")
            lines.append("")

        return "\n".join(lines), roadmap

    def format_career_response(self, user_text: str) -> Tuple[str, pd.DataFrame]:
        program = self.find_program_from_text(user_text)

        if program is None:
            return (
                "Saya belum dapat menentukan program studi yang dimaksud untuk menampilkan prospek karier.",
                pd.DataFrame(),
            )

        career_names = self.get_career_names_from_program(program)
        lines = [f"Prospek karier yang relevan untuk {program['nama_program_studi']}:\n"]
        career_rows = []

        if not career_names:
            lines.append("- Data prospek karier belum tersedia.")
        else:
            for career_name in career_names:
                matched = self.career_df[self.career_df["nama_karier"] == career_name]
                if matched.empty:
                    lines.append(f"- {career_name}")
                    career_rows.append({"nama_karier": career_name})
                else:
                    career = matched.iloc[0]
                    lines.append(f"- {career['nama_karier']} ({career['bidang_karier']})")
                    lines.append(f"  Tugas utama: {career['tugas_utama']}")
                    lines.append(f"  Skill teknis: {career['skill_teknis']}")
                    career_rows.append(career.to_dict())

        return "\n".join(lines), pd.DataFrame(career_rows)

    def format_skill_response(self, user_text: str) -> Tuple[str, pd.DataFrame]:
        program = self.find_program_from_text(user_text)

        if program is None:
            return (
                "Saya belum dapat menentukan program studi yang dimaksud untuk menampilkan skill awal.",
                pd.DataFrame(),
            )

        skill_names = self.get_skill_names_from_program(program)
        lines = [f"Skill awal yang disarankan untuk {program['nama_program_studi']}:\n"]
        skill_rows = []

        if not skill_names:
            lines.append("- Data skill awal belum tersedia.")
        else:
            for skill_name in skill_names:
                matched = self.skill_df[self.skill_df["nama_skill"] == skill_name]
                if matched.empty:
                    lines.append(f"- {skill_name}")
                    skill_rows.append({"nama_skill": skill_name})
                else:
                    skill = matched.iloc[0]
                    lines.append(f"- {skill['nama_skill']}")
                    lines.append(f"  Deskripsi: {skill['deskripsi_singkat']}")
                    lines.append(f"  Estimasi belajar: {skill['estimasi_waktu_belajar']}")
                    lines.append(f"  Sumber belajar awal: {skill['sumber_belajar_awal']}")
                    skill_rows.append(skill.to_dict())

        return "\n".join(lines), pd.DataFrame(skill_rows)

    def format_program_info_response(self, user_text: str) -> Tuple[str, pd.DataFrame]:
        program = self.find_program_from_text(user_text)

        if program is None:
            return (
                "Saya belum dapat menemukan program studi yang dimaksud. "
                "Coba sebutkan nama program studi, misalnya Teknik Informatika, Sistem Informasi, Sains Data, Statistika, atau DKV.",
                pd.DataFrame(),
            )

        career_names = self.get_career_names_from_program(program)
        skill_names = self.get_skill_names_from_program(program)

        lines = []
        lines.append(f"Informasi singkat tentang {program['nama_program_studi']} ({program['jenjang']}):\n")
        lines.append(f"- Rumpun ilmu: {safe_text(program.get('rumpun_ilmu'))}")
        lines.append(f"- Bidang keilmuan: {safe_text(program.get('bidang_keilmuan'))}")
        lines.append(f"- Deskripsi: {safe_text(program.get('deskripsi_singkat'))}")
        lines.append(f"- Minat yang cocok: {safe_text(program.get('minat_cocok'))}")
        lines.append(f"- Mata pelajaran relevan: {safe_text(program.get('mapel_relevan'))}")
        lines.append(f"- Prospek karier: {', '.join(career_names) if career_names else '-'}")
        lines.append(f"- Skill awal: {', '.join(skill_names) if skill_names else '-'}")

        return "\n".join(lines), pd.DataFrame([program.to_dict()])

    @staticmethod
    def format_clarification_response() -> str:
        return (
            "Agar saya bisa memberikan rekomendasi yang lebih tepat, coba jawab beberapa poin berikut:\n"
            "1. Mata pelajaran apa yang paling Anda sukai?\n"
            "2. Aktivitas atau hobi apa yang paling sering Anda lakukan?\n"
            "3. Apakah Anda lebih suka belajar dengan praktik, membaca teori, berdiskusi, atau membuat karya visual?\n"
            "4. Apakah Anda punya gambaran karier yang diinginkan?"
        )

    def chatbot_response(self, user_input: str, top_n: int = 3, return_dict: bool = False) -> Any:
        user_input = str(user_input).strip()

        if not user_input:
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_input": user_input,
                "final_intent": "empty_input",
                "routing_source": "validation",
                "model_predicted_intent": None,
                "model_confidence": 0.0,
                "rule_intent": None,
                "rule_confidence": 0.0,
                "matched_rule_keywords": "",
                "processed_text": "",
                "response_text": "Mohon tuliskan minat, hobi, mata pelajaran favorit, gaya belajar, atau tujuan karier Anda terlebih dahulu.",
                "payload": pd.DataFrame(),
            }
            return result if return_dict else result["response_text"]

        routing = self.route_intent(user_input)
        final_intent = routing["final_intent"]
        payload: Optional[pd.DataFrame] = None

        if final_intent == "sapaan":
            response_text = (
                "Halo, saya EduPath Career Assistant. "
                "Saya bisa membantu memberikan rekomendasi program studi S1, roadmap belajar awal, "
                "prospek karier, dan skill awal berdasarkan minat Anda. "
                "Silakan ceritakan minat, mapel favorit, hobi, gaya belajar, atau tujuan karier Anda."
            )
        elif final_intent == "rekomendasi_prodi":
            response_text, payload = self.format_recommendation_response(user_input, top_n=top_n)
        elif final_intent == "roadmap_belajar":
            response_text, payload = self.format_roadmap_response(user_input)
        elif final_intent == "prospek_karier":
            response_text, payload = self.format_career_response(user_input)
        elif final_intent == "skill_awal":
            response_text, payload = self.format_skill_response(user_input)
        elif final_intent == "info_program_studi":
            response_text, payload = self.format_program_info_response(user_input)
        elif final_intent == "klarifikasi_minat":
            response_text = self.format_clarification_response()
            payload = pd.DataFrame()
        else:
            response_text = (
                "Maaf, saya belum memahami pertanyaan Anda. "
                "Coba jelaskan minat, mata pelajaran favorit, hobi, gaya belajar, atau tujuan karier Anda. "
                "Contoh: 'Saya suka matematika dan data, program studi apa yang cocok?'"
            )
            payload = pd.DataFrame()

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_input": user_input,
            "final_intent": final_intent,
            "routing_source": routing["routing_source"],
            "model_predicted_intent": routing["model_result"]["predicted_intent"],
            "model_confidence": routing["model_result"]["confidence"],
            "top_intents": routing["model_result"].get("top_intents", []),
            "rule_intent": routing["rule_result"]["rule_intent"],
            "rule_confidence": routing["rule_result"]["rule_confidence"],
            "matched_rule_keywords": ", ".join(routing["rule_result"]["matched_keywords"]),
            "processed_text": routing["model_result"]["processed_text"],
            "response_text": response_text,
            "payload": payload if payload is not None else pd.DataFrame(),
        }

        return result if return_dict else response_text

    # ============================================================
    # Summary for Streamlit sidebar
    # ============================================================

    def get_system_summary(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.paths.project_root),
            "data_dir": str(self.paths.data_dir),
            "model_dir": str(self.paths.model_dir),
            "dataset_rows": {name: int(len(df)) for name, df in self.datasets.items()},
            "intent_model": type(self.intent_model).__name__,
            "intent_vectorizer": type(self.intent_vectorizer).__name__,
            "recommender_vectorizer": type(self.recommender_vectorizer).__name__,
            "recommender_matrix_shape": tuple(self.recommender_matrix.shape),
            "best_model_name": self.intent_metadata.get("best_model_name"),
            "recommender_method": self.recommender_config.get("method"),
            "intent_labels": self.intent_metadata.get("labels", []),
        }


# ============================================================
# Module-level Streamlit wrapper
# ============================================================

_CHATBOT_INSTANCE: Optional[EduPathChatbot] = None


def get_chatbot(
    project_root: Optional[Path] = None,
    force_reload: bool = False
) -> EduPathChatbot:
    """
    Return a singleton EduPathChatbot instance.

    This wrapper is intentionally placed at module level so app.py can import:
        from chatbot_utils import chatbot_response

    Parameters
    ----------
    project_root:
        Optional project root path. Normally not needed when running:
        streamlit run app/app.py
    force_reload:
        Set True when the app needs to reload datasets/models from disk.
    """
    global _CHATBOT_INSTANCE

    if force_reload or _CHATBOT_INSTANCE is None:
        _CHATBOT_INSTANCE = EduPathChatbot(project_root=project_root)

    return _CHATBOT_INSTANCE


def chatbot_response(
    user_text: Optional[str] = None,
    top_n: int = 3,
    return_dict: bool = False,
    query: Optional[str] = None,
    **_: Any
) -> Any:
    """
    Main chatbot function used by Streamlit app.py.

    This function delegates processing to EduPathChatbot.chatbot_response().
    It accepts both user_text and query so it remains compatible with
    different calling styles in app.py.

    Examples
    --------
    chatbot_response("Saya suka matematika dan komputer")
    chatbot_response(query="Saya ingin jadi Data Analyst")
    """
    final_text = user_text if user_text is not None else query

    if final_text is None:
        final_text = ""

    bot = get_chatbot()
    return bot.chatbot_response(
        user_input=str(final_text),
        top_n=top_n,
        return_dict=return_dict
    )
