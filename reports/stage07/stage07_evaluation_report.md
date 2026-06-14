# Stage 07 — Chatbot Evaluation, Error Analysis & Scenario Testing

Generated at: 2026-06-14 21:00:23

## 1. Ringkasan Metrik

                   metric   value
          total_scenarios 30.0000
          intent_accuracy  0.8667
response_content_accuracy  0.8667
     top_program_accuracy  0.9500
        overall_pass_rate  0.8667
         strict_pass_rate  0.8667
       confidence_ok_rate  1.0000

## 2. Ringkasan Per Intent

   expected_intent  total_cases  correct_intent  intent_accuracy  overall_pass_rate
          fallback            3               2           0.6667             0.6667
info_program_studi            3               3           1.0000             1.0000
 klarifikasi_minat            2               0           0.0000             0.0000
    prospek_karier            3               3           1.0000             1.0000
 rekomendasi_prodi           13              12           0.9231             0.9231
   roadmap_belajar            2               2           1.0000             1.0000
            sapaan            2               2           1.0000             1.0000
        skill_awal            2               2           1.0000             1.0000

## 3. Error Analysis

test_id     scenario_group   expected_intent      final_intent                                                                 error_type error_severity
  TC004 rekomendasi_formal rekomendasi_prodi          fallback Intent Routing Error; Recommendation Mapping Error; Response Content Error           High
  TC023  klarifikasi_minat klarifikasi_minat rekomendasi_prodi                               Intent Routing Error; Response Content Error           High
  TC024  klarifikasi_minat klarifikasi_minat rekomendasi_prodi                               Intent Routing Error; Response Content Error           High
  TC026           fallback          fallback rekomendasi_prodi               Fallback Error; Intent Routing Error; Response Content Error           High

## 4. Figure Output

- `evaluation_status_distribution.png` → `reports\stage07\figures\evaluation_status_distribution.png`
- `intent_accuracy_by_intent.png` → `reports\stage07\figures\intent_accuracy_by_intent.png`
- `intent_confusion_matrix.png` → `reports\stage07\figures\intent_confusion_matrix.png`
- `error_type_distribution.png` → `reports\stage07\figures\error_type_distribution.png`
- `routing_confidence_by_source.png` → `reports\stage07\figures\routing_confidence_by_source.png`

## 5. Catatan Evaluasi

Evaluasi ini merupakan evaluasi fungsional berbasis skenario. Hasil evaluasi perlu dibaca bersama dengan keterbatasan dataset, jumlah program studi yang masih terbatas, dan cakupan normalisasi bahasa yang masih baseline.