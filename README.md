# Instacart Market Basket Analysis 🛒📊

> **A Data Mining Project exploring Association Rules, Sequential Patterns, and Reorder Prediction on 3 Million Grocery Orders.**

## 📌 Project Overview
This project analyzes the [Instacart Market Basket Analysis (2017)](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset to uncover latent user preferences and predict future purchasing behavior. By leveraging techniques from **Frequent Itemset Mining**, **Sequential Pattern Mining**, and **Supervised Learning**, we aim to answer questions like:
*   *Do users buy Milk then Cereal, or do they buy Cereal then come back next week for Milk?*
*   *Can we predict exactly which items a user will reorder in their next basket with >40% F1 Score?*

## 📂 Dataset
*   **Source**: [Instacart (Kaggle Official)](https://www.instacart.com/datasets/grocery-shopping-2017)
*   **Size**: ~3.4 Million orders, 200k+ Users, 50k+ Products.
*   **Structure**: Relational tables (`orders`, `products`, `aisles`, `depts`) requiring complex joins.
*   **Key Characteristics**:
    *   **High Sparsity**: User-Item matrix density is <0.1%.
    *   **Strong Seasonality**: Distinct weekly purchase cycles (peaks at 7, 14, 21 days).
    *   **Extreme Class Imbalance**: "Bananas" dominate, while the long tail of products is vast.

## 🚀 Key Insights (so far)
From our initial Exploratory Data Analysis (`project_initiation.ipynb`), we have formulated three core hypotheses:

1.  **The Sparsity-Support Trade-off**:
    *   *Observation*: >90% of items appear in <1% of baskets.
    *   *Hypothesis*: Standard support thresholds (>0.01) will miss meaningful patterns in niche categories. We need **category-specific thresholds**.

2.  **Temporal Structure vs. Static Association**:
    *   *Observation*: Strong peaks in `days_since_prior_order` at 7-day intervals.
    *   *Hypothesis*: **Sequential Patterns** (Item A $\rightarrow$ Item B next week) may reveal structure missed by static Frequent Itemsets.

3.  **Predictability of Reorders**:
    *   *Observation*: Top 1% of products account for >20% of volume (Pareto Principle).
    *   *Hypothesis*: Global baselines are accurate but useless. Predictive models must rely on **User-Specific History** to be effective.

## 🛠️ Repository Structure
```bash
├── project_initiation.ipynb   # Part (A-E): Dataset Selection, EDA, and Hypothesis Formulation
├── kaggleInstacart/           # (Excluded) Raw data files
└── README.md                  # Project documentation
```

## 💻 How to Run
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/khussalpradhan/Instacart-Analysis.git
    cd Instacart-Analysis
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas matplotlib seaborn numpy
    ```
3.  **Download Data**:
    *   Download the dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
    *   Extract files into a `kaggleInstacart/` folder in the root directory.
4.  **Run the Notebook**:
    ```bash
    jupyter notebook project_initiation.ipynb
    ```

## 📬 Contact
*   **Author**: [Your Name/Pradh]
*   **Course**: Data Mining & Analysis (CSCE 670) at Texas A&M University
