# Dicoding Submission Checklist

Maps every requirement of *Proyek Akhir: Sistem Rekomendasi* (Belajar Machine Learning Terapan) to where it is satisfied.

## Submission files (mandatory)

| Requirement | Status | Evidence |
|---|:---:|---|
| `.zip` containing the three files | ✅ | `python scripts/build_submission.py` → `dist/submission_movie_recommendation.zip` |
| Markdown report (`.md`) | ✅ | [`laporan_proyek_machine_learning.md`](../laporan_proyek_machine_learning.md) |
| Python file (`.py`) | ✅ | [`sistem_rekomendasi_film.py`](../sistem_rekomendasi_film.py) (exported from the notebook; code identical to its cells, checked by the build script; runs standalone) |
| Jupyter notebook (`.ipynb`) **already executed** | ✅ | [`notebooks/sistem_rekomendasi_film.ipynb`](../notebooks/sistem_rekomendasi_film.ipynb): all 46 code cells executed, 0 errors; imports nothing from the repo and runs alone in an empty folder |
| Every code step documented with text cells | ✅ | 37 markdown cells (Indonesian) explain each step and its insights |
| Free dataset usable for recommendation | ✅ | MovieLens Latest Small (GroupLens) |
| Solution uses Content-Based **or** Collaborative Filtering | ✅ | Both, see Modeling |
| Original work, not previously submitted/published | ✅ | Authored for this submission |
| Report images load in Markdown | ✅ | Relative paths `outputs/figures/*.png`, bundled in the zip |

## Report rubric

| Category | Mandatory criterion | Where | Additional criterion (⭐) | Where |
|---|---|---|---|---|
| **Project Overview** | Relevant background | Report → *Latar belakang* | Why the problem matters and how to solve it | *Mengapa dan bagaimana masalah ini diselesaikan* ✅ |
| | | | Research / credible references | 11 APA references with DOIs ✅ |
| **Business Understanding** | Problem statements | 4 problem statements | *Solution Approach* with 2 approaches (CBF + CF) | *Solution Approach* table (CBF, SVD, NeuMF) ✅ |
| | Goals | 4 goals, one per problem statement | | |
| **Data Understanding** | Data size, condition, info | *Informasi dataset* + table | EDA with visualizations **and insights** | 8 figures, each with an insight ✅ |
| | Download link | GroupLens page + direct zip link | | |
| | All variables described | All 13 columns across 4 files | | |
| **Data Preparation** | Techniques applied and listed | 7 ordered steps | Process explained | Each step has *what* + code where useful ✅ |
| | Same order as notebook | Report 1–7 = notebook 4.1–4.7 | Reason for each step | Each step has an **Alasan** paragraph ✅ |
| **Modeling & Result** | Recommender explained | CBF, SVD, NeuMF sections | Two solutions with different algorithms | Three algorithms ✅ |
| | Top-N output | Top-10 tables for CBF (Toy Story), SVD and NeuMF (user 1) | Pros & cons of each approach | **Kelebihan / Kekurangan** per model ✅ |
| **Evaluation** | Metrics named | Precision@K, Recall@K, NDCG@K, Coverage, RMSE, MAE | Formulas and how the metrics work | LaTeX formulas + worked examples ✅ |
| | Results explained with metrics | Two result tables + interpretation + conclusion | | |
| **Report structure** | Follows the template, organized | Template headings kept in order | | |
| | Code snippets only where needed | 4 short snippets | | |
| | Images render in Markdown | Relative image paths | | |

**Additional criteria met: 6 / 6** (Project Overview, Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation), which targets ⭐⭐⭐⭐⭐.

## Self-review checklist (Review Mandiri)

- [x] Zip with 3 files: `.md` report, `.py` file, executed `.ipynb`
- [x] Every code cell documented with text cells
- [x] Free dataset usable for recommendations
- [x] Solution uses Content-Based Filtering / Collaborative Filtering
- [x] Report meets the **Project Overview** rubric
- [x] Report meets the **Business Understanding** rubric
- [x] Report meets the **Data Understanding** rubric
- [x] Report meets the **Data Preparation** rubric
- [x] Report meets the **Modeling and Results** rubric
- [x] Report meets the **Evaluation** rubric
- [x] Report meets the **Report Structure** rubric

## Before uploading

1. Re-run the notebook top to bottom (Kernel → Restart & Run All) if any code changed.
2. Re-export the script: `jupyter nbconvert --to script notebooks/sistem_rekomendasi_film.ipynb --output-dir .`
3. Build the zip: `python scripts/build_submission.py`
4. Open the zip and check the report renders with its images (e.g. VS Code Markdown preview).
