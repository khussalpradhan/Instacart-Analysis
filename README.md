# Instacart Market Basket Analysis: Sequential & Associative Pattern Mining

> **A Data Mining Project combining FP-Growth and PrefixSpan to uncover the hidden temporal rhythm of grocery shopping.**

**Author:** Khussal Pradhan · UID: 437005859  
**Course:** CSCE 670 — Data Mining & Analysis · Texas A&M University · Spring 2026

👉 **Start here:** `main_notebook.ipynb`

---

## 1. Project Overview
Online grocery platforms generate vast transactional datasets that encode rich behavioral signals. Standard recommendation engines usually look for items bought *together* in a single trip, essentially ignoring time. This project analyzes a reproducible sample of Instacart's massive dataset to compare traditional static co-occurrence (using **FP-Growth**) against longitudinal, temporal pattern mining (using **PrefixSpan**). The goal is to detect the hidden temporal rhythms of grocery shopping—predicting not just what else you might buy today, but what you will buy *next week*.

## 2. Main Deliverable
👉 **The main deliverable is `main_notebook.ipynb`**

## 3. Research Questions
1. **RQ1 (FP-Growth Threshold Sweep):** What frequent itemsets and association rules emerge under varying support thresholds, and how do confidence and lift compare in an extremely sparse space?
2. **RQ2 (Segmented Mining):** How do purchasing patterns differ between early-week (Days 0-1) and late-week (Days 5-6) shoppers?
3. **RQ3 (PrefixSpan vs Static Rules):** Do sequential patterns reveal multi-step item trajectories across consecutive orders that static association rules structurally miss?

## 4. Project Video
🎥 **[Watch the 2-Minute Project Pitch on YouTube](https://www.youtube.com/watch?v=G2PMk_WyP-Q)**

## 5. Data Section
- **Dataset Source:** [Instacart Market Basket Analysis (2017) via Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
- **Scope:** The raw dataset contains 3.4 million orders across 200,000+ users. 
- **Preprocessing:** Due to computational constraints, this project initially analyzes a deterministic, reproducible sample of 5,000 users. FP-Growth was run on the entire 5,000-user sample. PrefixSpan was run on a computationally constrained filtered subset: the first 2,000 sampled users and top 20 products, yielding 1,671 users after filtering. Relational tables were joined to extract longitudinal, chronological order histories per user, maintaining exact basket boundaries.

## 6. How to Reproduce
This project was built and executed in Google Colab.
1. Clone this repository.
2. Download the Kaggle dataset files and place them in a `kaggleInstacart/` folder at the root of the repository.
3. Install the specific environment packages via `pip install -r requirements.txt`.
4. Run the notebooks in the following order:
   - `checkpoints/checkpoint_1.ipynb` (Initial EDA)
   - `checkpoints/checkpoint_2.ipynb` (Feasibility)
   - `main_notebook.ipynb` (Final curated execution and results)

## 7. Key Dependencies
This notebook runs on Python 3.10+ in a standard Colab environment. Key libraries used:
- `pandas` (2.0.3)
- `numpy` (1.25.2)
- `mlxtend` (0.23.1) — For FP-Growth and Association Rules
- `prefixspan` (0.5.2) — For Sequential Pattern Mining

*See `requirements.txt` for the full environment export.*

## 8. Repository Structure
```
.
├── main_notebook.ipynb             # FINAL CURATED PROJECT — Start Here!
├── README.md                       # Project documentation
├── requirements.txt                # Colab environment package versions
├── checkpoints/                    # Progression of work over the semester
│   ├── checkpoint_1.ipynb          # Dataset selection & base EDA
│   └── checkpoint_2.ipynb          # RQ formulation & feasibility testing
├── scripts/                        # Data processing / generator scripts
│   ├── generate_nb.py              
│   ├── generate_final_nb.py        
│   └── test_ps.py                  
├── assets/                         # Pitch materials and generated assets
│   ├── pitch_slides.html           
│   └── pitch_slides.pdf            
└── kaggleInstacart/                # (Ignored) Raw CSV data
```

## 9. Results Summary
Our analysis uncovered that grocery data exhibits extreme matrix sparsity (>99.98%), requiring support thresholds as low as 0.5% for meaningful associations to emerge. At this threshold, FP-Growth discovered **344 significant static association rules**, largely dominated by organic produce clusters. 

However, standard co-occurrence analysis is structurally blind to time. By structuring the data chronologically, PrefixSpan captured thousands of repeated temporal purchase patterns, including recurring staple-purchase sequences across multiple orders. These patterns show that sequential order history adds information beyond static basket co-occurrence.
