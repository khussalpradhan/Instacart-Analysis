# Instacart Market Basket Analysis: Sequential & Associative Pattern Mining

> **A Data Mining Project combining FP-Growth and PrefixSpan to uncover the hidden temporal rhythm of grocery shopping across 3.4 Million orders.**

**Author:** Khussal Pradhan · UID: 437005859  
**Course:** CSCE 670 — Data Mining & Analysis · Texas A&M University · Spring 2026

---

## Project Overview

This project analyzes the [Instacart Market Basket Analysis (2017)](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset to uncover latent purchasing patterns using two complementary techniques:

- **FP-Growth** (Course Technique) — mines intra-basket association rules to find products frequently bought *together*
- **PrefixSpan** (External Technique) — mines inter-basket sequential patterns to find products bought *in sequence* across orders over time

### Key Results

| Metric | Value |
|--------|-------|
| Association Rules (FP-Growth @ 0.5% support) | **344** |
| Sequential Patterns (PrefixSpan) | **10,046** |
| Temporal-only Item Pairs | **133** |
| Research Questions Fully Answered | **3** |

The project's key finding: **133 item pairs** are bought in *sequence* across orders but *never* in the same basket (e.g., Organic Raspberries → Organic Zucchini). FP-Growth is structurally blind to these temporal dependencies — only PrefixSpan can detect them.

## Dataset
- **Source**: [Instacart (Kaggle Official)](https://www.instacart.com/datasets/grocery-shopping-2017)
- **Size**: ~3.4 Million orders, 200K+ Users, 50K+ Products
- **Structure**: Relational tables (`orders`, `products`, `aisles`, `departments`)
- **Key Characteristics**:
    - **>99.9% Sparsity** in the user-product matrix
    - **Strong weekly cycles** — reorder peaks at 7, 14, 21, and 30 days
    - **Extreme long-tail** — most products appear in <1% of baskets

## Research Questions

1. **RQ1 (FP-Growth Threshold Sweep):** What frequent itemsets and association rules emerge under varying support thresholds (0.1% to 5%), and how do confidence and lift compare?
2. **RQ2 (Segmented Mining):** How do purchasing patterns differ between early-week (Days 0-1) and late-week (Days 5-6) shoppers?
3. **RQ3 (PrefixSpan vs Static Rules):** Do sequential patterns reveal item dependencies across orders that static association rules miss?

## Repository Structure
```
├── instacart_final_project.ipynb     # FINAL PROJECT — Full analysis & results
├── project_checkpoint_2.ipynb        # Checkpoint 2: RQ formulation & feasibility
├── project_initiation_FINAL.ipynb    # Checkpoint 1: Dataset selection & base EDA
├── pitch_slides.html                 # 5-slide investor pitch deck
├── kaggleInstacart/                  # (Excluded via .gitignore) Raw data files
└── README.md                         # This file
```

## How to Run
1. **Clone the repository**:
    ```bash
    git clone https://github.com/khussalpradhan/Instacart-Analysis.git
    cd Instacart-Analysis
    ```
2. **Install dependencies**:
    ```bash
    pip install pandas matplotlib seaborn numpy mlxtend prefixspan nbformat jupyter
    ```
3. **Download Data**:
    - Download from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
    - Extract files into a `kaggleInstacart/` folder in the root directory
4. **Run the Final Notebook**:
    ```bash
    jupyter notebook instacart_final_project.ipynb
    ```

## Deliverables
| Deliverable | Description |
|-------------|-------------|
| `instacart_final_project.ipynb` | Complete analysis notebook (15 code cells, 16 markdown, 9 plots) |
| `pitch_slides.html` | 5-slide pitch deck — open in browser for fullscreen presentation |
| Video (YouTube) | 2-minute investor pitch |

## Contact
- **Author**: Khussal Pradhan (UID: 437005859)
- **Course**: Data Mining & Analysis (CSCE 670) at Texas A&M University
