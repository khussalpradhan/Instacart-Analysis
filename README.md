# Instacart Market Basket Analysis 

> **A Data Mining Project exploring Association Rules, Sequential Patterns, and Reorder Prediction on 3 Million Grocery Orders.**

## Project Overview
This project analyzes the [Instacart Market Basket Analysis (2017)](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset to uncover latent user preferences and predict future purchasing behavior. By leveraging techniques from **Frequent Itemset Mining**, **Sequential Pattern Mining**, and **Supervised Learning**, we aim to answer questions like:
*   *Do users buy Milk then Cereal, or do they buy Cereal then come back next week for Milk?*
*   *Can we predict exactly which items a user will reorder in their next basket with >40% F1 Score?*

## Dataset
*   **Source**: [Instacart (Kaggle Official)](https://www.instacart.com/datasets/grocery-shopping-2017)
*   **Size**: ~3.4 Million orders, 200k+ Users, 50k+ Products.
*   **Structure**: Relational tables (`orders`, `products`, `aisles`, `depts`) requiring complex joins.
*   **Key Characteristics**:
    *   **High Sparsity**: User-Item matrix density is <0.1%.
    *   **Strong Seasonality**: Distinct weekly purchase cycles (peaks at 7, 14, 21 days).
    *   **Extreme Class Imbalance**: "Bananas" dominate, while the long tail of products is vast.

## Research Questions (Checkpoint 2)
Building upon our initial EDA, we have formulated three core research questions combining Course and External data mining techniques:

1.  **RQ1 (Course - FP-Growth):** What frequent itemsets and association rules emerge under varying support thresholds, and how do confidence and lift compare when evaluating these rules?
    *   *Feasibility:* EDA proves a highly sparse long-tail distribution, requiring mathematical justification for extremely low support thresholds (e.g., 0.001 - 0.01).
2.  **RQ2 (Course - Segmented FP-Growth):** How do frequent purchasing patterns differ between distinct user segments (e.g., early-week vs. late-week shoppers)?
    *   *Feasibility:* EDA segmentation reveals massive shopping volume variances between early-week (Days 0, 1) and late-week shoppers, allowing for comparative rule mining.
3.  **RQ3 (External - PrefixSpan):** Do sequential purchase patterns (e.g., buying Basket A, followed by Basket B in a subsequent order) reveal item dependency structures across orders missed by intra-basket unordered itemsets?
    *   *Feasibility:* Restructuring order histories into alphabetical and hashable basket tuples proves mathematically sound PrefixSpan sequence tracking. While memory intensive, running PrefixSpan on our randomly sampled user subsets will be technically feasible without artificially truncating user histories.

## Repository Structure
```bash
├── project_checkpoint_2.ipynb       # Checkpoint 2: Research Questions & Methodological Feasibility EDA
├── project_initiation_FINAL.ipynb   # Checkpoint 1: Dataset Selection, Base EDA, and Hypothesis Formulation
├── kaggleInstacart/                 # (Excluded) Raw data files (.csv)
└── README.md                        # Project documentation
```

## How to Run
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/khussalpradhan/Instacart-Analysis.git
    cd Instacart-Analysis
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas matplotlib seaborn numpy mlxtend prefixspan nbformat jupyter
    ```
3.  **Download Data**:
    *   Download the dataset from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data).
    *   Extract files into a `kaggleInstacart/` folder in the root directory.
4.  **Run the Notebooks**:
    ```bash
    jupyter notebook project_initiation_FINAL.ipynb
    jupyter notebook project_checkpoint_2.ipynb
    ```

## Contact
*   **Author**: Khussal Pradhan
*   **Course**: Data Mining & Analysis (CSCE 670) at Texas A&M University
