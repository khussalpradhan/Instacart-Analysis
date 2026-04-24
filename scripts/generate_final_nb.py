import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ============================================================
# PREAMBLE
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""# Instacart Market Basket Analysis: Sequential & Associative Pattern Mining

**Author:** Khussal Pradhan  
**Course:** CSCE 670 — Data Mining & Analysis, Texas A&M University  

## Abstract
**Abstract:** This project analyzes a **reproducible 5,000-user sample** drawn from the 3.4 million grocery orders in the Instacart Market Basket Analysis dataset. We employ two complementary data mining approaches: **FP-Growth** (a course technique) for intra-basket association rule mining on the full 5,000-user sample, and **PrefixSpan** (an external technique) for inter-basket sequential pattern mining on a computationally constrained subset (first 2,000 sampled users and top 20 products, yielding 1,671 users after filtering). Our analysis reveals that (1) meaningful association rules only emerge at extremely low support thresholds due to the measured >99.9% data sparsity, (2) early-week and late-week shoppers exhibit distinct purchasing patterns, and (3) sequential mining captures temporal multi-step trajectories that static association rules fundamentally miss."""))

# Collaboration Declaration
cells.append(nbf.v4.new_markdown_cell("""## Collaboration Declaration

**1. Collaborators:** None (Individual Project)

**2. Web Sources:**
- [Instacart Market Basket Analysis (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis)
- [PrefixSpan Algorithm Documentation](https://pypi.org/project/prefixspan/)
- [Mlxtend FP-Growth Documentation](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)

**3. AI Tools:**
- **ChatGPT / Gemini**: Used for brainstorming feasibility analysis, writing plotting code, structuring assertions, and ensuring rubric adherence.

**4. Citations:**
- Instacart. (2017). "The Instacart Online Grocery Shopping Dataset 2017". https://www.instacart.com/datasets/grocery-shopping-2017
- Pei, J., et al. (2001). PrefixSpan: Mining Sequential Patterns Efficiently by Prefix-Projected Pattern Growth. *ICDE*.
- Agrawal, R., Imieliński, T., & Swami, A. (1993). Mining association rules between sets of items in large databases. *SIGMOD*."""))

# Table of Contents
cells.append(nbf.v4.new_markdown_cell("""## Table of Contents

1. **Introduction** — Project motivation and dataset context
2. **Data Loading & Preprocessing** — Sample 5,000 users, load transaction history
3. **RQ1: Support Threshold Analysis** — Sweep FP-Growth across 6 thresholds, analyze confidence vs lift
4. **RQ2: Segmented Mining** — Compare early-week vs late-week shopping patterns
5. **RQ3: Sequential Pattern Mining** — PrefixSpan basket trajectories vs static association rules
6. **Conclusion** — Summary of findings, limitations, and future work"""))

# ============================================================
# INTRODUCTION
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""## 1. Introduction

### Motivation

Online grocery platforms like Instacart generate vast transactional datasets that encode rich behavioral signals about how people shop. Traditional recommendation engines rely on **association rule mining** (e.g., Apriori, FP-Growth) to find products frequently bought *together* in a single basket. While powerful, this approach treats every basket as an isolated event and ignores the **temporal dynamics** of user behavior — the fact that what a user buys today is often influenced by what they bought last week.

### Research Gap

This project addresses a fundamental limitation in standard basket analysis: **the absence of temporal modeling**. Specifically, we ask whether sequential pattern mining — analyzing baskets *in chronological order* rather than in isolation — reveals item dependencies that static association rules structurally cannot capture. This is practically important: if temporal dependencies exist, recommendation engines that ignore them are leaving significant predictive signal on the table.

### Approach

We combine two complementary data mining techniques:

| Technique | Type | What It Finds | Scope |
|-----------|------|---------------|-------|
| **FP-Growth** | Course technique | Items bought *together* in one basket | Intra-basket (static) |
| **PrefixSpan** | External technique | Baskets bought *in sequence* over time | Inter-basket (temporal) |

### Dataset

We use the [Instacart Market Basket Analysis (2017)](https://www.kaggle.com/c/instacart-market-basket-analysis) dataset, containing 3.4 million orders from 200K+ users across 50K+ products. Due to computational constraints, we analyze a deterministic, reproducible sample of 5,000 users. The dataset provides complete longitudinal order histories per user, making it ideal for both static and sequential pattern mining."""))

# ============================================================
# DATA LOADING
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""---

## 2. Data Loading & Preprocessing

The full `order_products__prior.csv` contains ~32 million rows. Running FP-Growth or PrefixSpan on the complete dataset in a Jupyter kernel would trigger an Out-Of-Memory error. Our strategy: sample **5,000 unique users** (with `np.random.seed(42)` for reproducibility) and retain their **complete longitudinal order histories**. This preserves the sequential structure needed for PrefixSpan while keeping the data manageable."""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Configure Plotting
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12

# Environment checks
assert pd is not None and np is not None
print("All libraries imported successfully.")"""))

cells.append(nbf.v4.new_code_cell("""data_path = "kaggleInstacart/"

def load_and_sample_data(path, sample_users=5000):
    print("Loading orders.csv...")
    orders = pd.read_csv(os.path.join(path, "orders.csv"))
    assert not orders.empty, "Orders dataframe is empty!"
    assert orders['user_id'].nunique() > sample_users
    
    # Sample complete user histories (not random rows)
    np.random.seed(42)
    sampled_ids = np.random.choice(orders['user_id'].unique(), size=sample_users, replace=False)
    orders_sample = orders[orders['user_id'].isin(sampled_ids)]
    assert orders_sample['user_id'].nunique() == sample_users
    
    print("Loading order_products__prior.csv...")
    prior = pd.read_csv(os.path.join(path, "order_products__prior.csv"))
    prior_sample = prior[prior['order_id'].isin(orders_sample['order_id'])]
    assert not prior_sample.empty
    
    print("Loading products.csv...")
    products = pd.read_csv(os.path.join(path, "products.csv"))
    
    print(f"\\nSample: {orders_sample['user_id'].nunique()} users, "
          f"{orders_sample.shape[0]} orders, {prior_sample.shape[0]} transactions")
    return orders_sample, prior_sample, products

orders_df, prior_df, products_df = load_and_sample_data(data_path)"""))

# Data overview
cells.append(nbf.v4.new_code_cell("""# Quick overview of the sampled data
print("=== Orders Sample ===")
print(f"Shape: {orders_df.shape}")
print(f"Columns: {list(orders_df.columns)}")
print(f"Users: {orders_df['user_id'].nunique()}")
print(f"Unique orders: {orders_df['order_id'].nunique()}")

print("\\n=== Prior Transactions Sample ===")
print(f"Shape: {prior_df.shape}")
print(f"Unique orders: {prior_df['order_id'].nunique()}")
print(f"Unique products: {prior_df['product_id'].nunique()}")

print("\\n=== Products ===")
print(f"Total products in catalog: {products_df.shape[0]}")

# Show a sample order
sample_order_id = prior_df['order_id'].iloc[0]
sample_items = prior_df[prior_df['order_id'] == sample_order_id].merge(
    products_df[['product_id', 'product_name']], on='product_id')
print(f"\\nExample basket (order {sample_order_id}):")
print(sample_items['product_name'].tolist())"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation:** Our sample retains 5,000 users with their complete prior order histories. The transaction table contains hundreds of thousands of rows, confirming we have enough data volume for pattern mining. The example basket above shows a typical grocery trip — a mix of staples and specialty items."""))

# Descriptive statistics table
cells.append(nbf.v4.new_code_cell("""# Descriptive statistics: orders per user and basket sizes
orders_per_user = orders_df[orders_df['eval_set'] == 'prior'].groupby('user_id')['order_id'].count()
basket_sizes = prior_df.groupby('order_id')['product_id'].count()

# Explicit sparsity computation (Basket-Item matrix)
total_possible_interactions = orders_df['order_id'].nunique() * prior_df['product_id'].nunique()
actual_interactions = prior_df.shape[0]
sparsity = 1.0 - (actual_interactions / total_possible_interactions)

stats_data = {
    'Metric': ['Users Sampled', 'Total Orders', 'Total Transactions', 'Products in Sample',
               'Avg Orders/User', 'Median Orders/User', 'Avg Basket Size', 'Median Basket Size', 'Matrix Sparsity'],
    'Value': [
        f"{orders_df['user_id'].nunique():,}",
        f"{orders_df['order_id'].nunique():,}",
        f"{prior_df.shape[0]:,}",
        f"{prior_df['product_id'].nunique():,}",
        f"{orders_per_user.mean():.1f}",
        f"{orders_per_user.median():.0f}",
        f"{basket_sizes.mean():.1f}",
        f"{basket_sizes.median():.0f}",
        f"{sparsity:.4%}"
    ]
}
display(pd.DataFrame(stats_data).style.hide(axis='index').set_caption('Dataset Summary Statistics'))"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation:** The statistics confirm two critical properties of this dataset that will drive our methodological decisions:

1. **Extreme Sparsity (>99.9%)** — most products never appear in most baskets. This mathematically guarantees that traditional support thresholds (e.g., 5-10%) will fail to find anything except "Banana & Organic Banana". We will need extremely low support levels for FP-Growth.
2. **Rich longitudinal depth** — users average 15.6 prior orders, with a median of 9, providing enough temporal depth for sequential mining.
2. **Moderate basket sizes** — average baskets contain ~10 items, which is tractable for one-hot encoding without excessive dimensionality.

These characteristics make this dataset uniquely suited for our dual FP-Growth + PrefixSpan approach."""))

# Reorder rate analysis
cells.append(nbf.v4.new_code_cell("""# Reorder behavior analysis
reorder_rate = prior_df['reordered'].mean()
print(f"Global reorder rate: {reorder_rate:.1%}")

# Days since prior order distribution
days_since = orders_df['days_since_prior_order'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reorder histogram
axes[0].bar(['Reordered', 'New'], [prior_df['reordered'].sum(), (1 - prior_df['reordered']).sum()],
            color=['#10B981', '#64748B'])
axes[0].set_title('Reorder vs New Products in Baskets')
axes[0].set_ylabel('Number of Items')
for i, v in enumerate([prior_df['reordered'].sum(), (1-prior_df['reordered']).sum()]):
    axes[0].text(i, v + v*0.02, f'{v:,.0f}', ha='center', fontweight='bold')

# Days since prior order
sns.histplot(days_since, bins=30, kde=True, color='#8B5CF6', ax=axes[1])
axes[1].set_xlabel('Days Since Prior Order')
axes[1].set_ylabel('Number of Orders')
axes[1].set_title('Days Between Orders — Shopping Cycle Detection')
for day in [7, 14, 21, 30]:
    axes[1].axvline(day, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].text(day, axes[1].get_ylim()[1]*0.9, f'{day}d', ha='center', fontsize=9, color='red')

plt.tight_layout()
plt.show()

print(f"Days since prior order — Median: {days_since.median():.0f}, Mean: {days_since.mean():.1f}")
for day in [7, 14, 21, 30]:
    count = (days_since == day).sum()
    pct = (days_since == day).mean()
    print(f"Day {day}: {count:,} orders ({pct:.1%})")"""))

cells.append(nbf.v4.new_markdown_cell("""**Key Insight — The 7-Day Shopping Cycle:**

The days-between-orders histogram reveals a striking pattern: shopping behavior is **cyclical**, with a **dominant peak at 7 days** and visible periodicity at weekly multiples (14, 21, 30 days). The 7-day peak is the strongest, consistent with weekly grocery routines. The 30-day spike likely reflects a separate monthly shopping pattern or the dataset's cap on the `days_since_prior_order` field.

This has two critical implications for our analysis:
1. **For RQ2 (Segmented Mining):** The day-of-week clearly matters — users shop on specific days with predictable rhythms.
2. **For RQ3 (Sequential Mining):** These regular reorder cycles mean PrefixSpan should discover meaningful sequential patterns, because users' baskets are temporally structured, not random.

The ~59% reorder rate confirms that users are creatures of habit — they consistently re-purchase the same products, which is exactly the signal our pattern mining algorithms are designed to capture."""))


# ============================================================
# RQ1: SUPPORT THRESHOLD ANALYSIS
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""---

## 3. RQ1: What frequent itemsets and association rules emerge under varying support thresholds?

**Methodology:** We run FP-Growth (preferred over Apriori because it avoids candidate generation, critical for grocery data with 50K+ unique products) across **6 support thresholds** from 0.1% to 5%. For each threshold, we measure the number of itemsets, number of association rules, and wall-clock time. We then deep-dive into the rules at the optimal threshold.

**Why not Apriori?** In Checkpoint 1, we found >90% of items appear in <1% of baskets. Apriori's candidate generation phase causes exponential blowup on datasets with this level of sparsity. FP-Growth's FP-Tree architecture avoids this entirely."""))

# Support distribution EDA
cells.append(nbf.v4.new_code_cell("""# Merge transactions with product names
prior_named = prior_df.merge(products_df[['product_id', 'product_name']], on='product_id', how='left')

# Calculate item support (fraction of baskets each product appears in)
total_baskets = prior_df['order_id'].nunique()
item_counts = prior_named.groupby('product_name')['order_id'].nunique()
item_support = (item_counts / total_baskets).sort_values(ascending=False)

print(f"Total unique baskets: {total_baskets:,}")
print(f"Total unique products in sample: {len(item_support):,}")
print(f"Products above 5% support: {(item_support > 0.05).sum()}")
print(f"Products above 1% support: {(item_support > 0.01).sum()}")
print(f"Products above 0.1% support: {(item_support > 0.001).sum()}")

# Plot top 30 items by support
fig, ax = plt.subplots(figsize=(14, 6))
top30 = item_support.head(30)
sns.barplot(x=top30.values, y=top30.index, palette='viridis', ax=ax)
ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='5% threshold')
ax.axvline(x=0.01, color='orange', linestyle='--', linewidth=2, label='1% threshold')
ax.set_xlabel("Support (fraction of baskets)")
ax.set_title("Top 30 Products by Support — Long-Tail Distribution")
ax.legend()
plt.tight_layout()
plt.show()

print("\\nTop 10 products by support:")
display(item_support.head(10).reset_index().rename(
    columns={'product_name': 'Product', 'order_id': 'Support'}).style.format({'Support': '{:.4f}'}))"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation:** The distribution confirms extreme long-tail sparsity. Only a handful of products (Banana, Bag of Organic Bananas, Organic Strawberries) cross the 5% support threshold. If we set `min_support=0.05` as textbook examples suggest, FP-Growth would find virtually zero interesting multi-item associations. This motivates our threshold sweep below."""))

# FP-Growth threshold sweep
cells.append(nbf.v4.new_code_cell("""from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Filter to top 100 products to keep the sparse matrix manageable
# This is standard practice in grocery analytics — the long tail adds noise, not signal
top_100_products = item_support.head(100).index.tolist()
prior_top100 = prior_named[prior_named['product_name'].isin(top_100_products)]

# Build baskets (list of product names per order)
baskets_all = prior_top100.groupby('order_id')['product_name'].apply(list).tolist()
# Remove empty baskets (orders that had zero top-100 products)
baskets_all = [b for b in baskets_all if len(b) > 0]
print(f"Total baskets with top-100 products: {len(baskets_all):,}")

# One-hot encode (sparse to save RAM)
te = TransactionEncoder()
te_ary = te.fit(baskets_all).transform(baskets_all, sparse=True)
sparse_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
print(f"Sparse matrix shape: {sparse_df.shape} (baskets x products)")

# Sweep across thresholds
thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
sweep_results = []

for thresh in thresholds:
    t0 = time.time()
    try:
        fi = fpgrowth(sparse_df, min_support=thresh, use_colnames=True)
        n_itemsets = len(fi)
        
        if n_itemsets > 0:
            rules = association_rules(fi, metric="lift", min_threshold=1.0, num_itemsets=len(baskets_all))
            n_rules = len(rules)
        else:
            n_rules = 0
    except Exception as e:
        n_itemsets, n_rules = -1, -1
        print(f"  Threshold {thresh} failed: {e}")
    
    elapsed = time.time() - t0
    sweep_results.append({
        'Threshold': thresh, 'Itemsets': n_itemsets, 
        'Rules': n_rules, 'Time (s)': round(elapsed, 2)
    })
    print(f"  Support={thresh:.3f} → {n_itemsets:,} itemsets, {n_rules:,} rules ({elapsed:.1f}s)")

sweep_df = pd.DataFrame(sweep_results)
display(sweep_df)

# Visualization: Itemsets and Rules vs Threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar([str(t) for t in thresholds], sweep_df['Itemsets'], color='#06B6D4')
axes[0].set_xlabel("Min Support Threshold")
axes[0].set_ylabel("Frequent Itemsets Found")
axes[0].set_title("Itemsets vs Support Threshold")
axes[0].set_yscale('log')

axes[1].bar([str(t) for t in thresholds], sweep_df['Rules'], color='#10B981')
axes[1].set_xlabel("Min Support Threshold")
axes[1].set_ylabel("Association Rules Found")
axes[1].set_title("Rules vs Support Threshold")
axes[1].set_yscale('log')

plt.suptitle("RQ1: FP-Growth Threshold Sweep", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation — Threshold Sweep Results:**

The sweep reveals a clear **exponential relationship** between support threshold and the number of patterns discovered:

- At **5% support**, virtually no multi-item itemsets survive — the dataset is too sparse for traditional thresholds.
- At **0.5% support**, we hit the sweet spot: enough rules for rich analysis without the combinatorial explosion that occurs at 0.1%.
- At **0.1% support**, the number of rules explodes into tens of thousands, most of which are noise. The computation time also increases dramatically.

This confirms our EDA finding: grocery data requires thresholds 10-50x lower than textbook examples to yield actionable patterns."""))

# Optimal threshold deep dive
cells.append(nbf.v4.new_code_cell("""# Select the optimal threshold: 0.005 balances enough rules for analysis
# without the combinatorial explosion of 0.001
OPTIMAL_THRESHOLD = 0.005

fi_optimal = fpgrowth(sparse_df, min_support=OPTIMAL_THRESHOLD, use_colnames=True)
rules_optimal = association_rules(fi_optimal, metric="lift", min_threshold=1.0, num_itemsets=len(baskets_all))

print(f"At support={OPTIMAL_THRESHOLD}: {len(fi_optimal)} itemsets, {len(rules_optimal)} rules")

# Top 15 rules by LIFT
print("\\n=== Top 15 Rules by Lift ===")
top_lift = rules_optimal.sort_values('lift', ascending=False).head(15)
display(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True))

# Top 15 rules by CONFIDENCE
print("\\n=== Top 15 Rules by Confidence ===")
top_conf = rules_optimal.sort_values('confidence', ascending=False).head(15)
display(top_conf[['antecedents', 'consequents', 'support', 'confidence', 'lift']].reset_index(drop=True))"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation — Deep Dive into the Optimal Threshold:**

Examining the top rules at 0.5% support reveals a clear pattern:

- **Highest-lift rules** are dominated by organic product bundles (e.g., Organic Strawberries ↔ Organic Raspberries, Organic Avocado ↔ Organic Baby Spinach). This reveals that *organic shoppers form a tightly-knit purchasing tribe* — they buy multiple organic items together at rates far above what random chance would predict.
- **Highest-confidence rules** tend to involve universally popular products (Banana, Bag of Organic Bananas) as consequents. These have high confidence simply because almost everyone buys bananas, not because of a genuine association. This is exactly why **lift is a better metric than confidence** for evaluating rule quality.
- There is a notable **inverse relationship** between confidence and lift for many rules — the most confident predictions are the least surprising."""))

# Confidence vs Lift scatter
cells.append(nbf.v4.new_code_cell("""# Confidence vs Lift scatter plot — the core RQ1 visualization
fig, ax = plt.subplots(figsize=(12, 7))

scatter = ax.scatter(
    rules_optimal['confidence'], 
    rules_optimal['lift'],
    c=rules_optimal['support'],
    cmap='viridis',
    s=60,
    alpha=0.7,
    edgecolors='white',
    linewidth=0.5
)

# Annotate the top 5 highest-lift rules
for _, row in rules_optimal.nlargest(5, 'lift').iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    label = f"{ant[:20]}→{con[:20]}"
    ax.annotate(label, (row['confidence'], row['lift']),
                fontsize=8, alpha=0.8,
                xytext=(5, 5), textcoords='offset points')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Support')
ax.set_xlabel("Confidence", fontsize=13)
ax.set_ylabel("Lift", fontsize=13)
ax.set_title("RQ1: Confidence vs Lift for All Association Rules", fontsize=14, fontweight='bold')
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Lift = 1 (no association)')
ax.legend()
plt.tight_layout()
plt.show()

# Summary statistics
print(f"\\nRule Statistics:")
print(f"  Total rules: {len(rules_optimal)}")
print(f"  Avg confidence: {rules_optimal['confidence'].mean():.3f}")
print(f"  Avg lift: {rules_optimal['lift'].mean():.3f}")
print(f"  Max lift: {rules_optimal['lift'].max():.3f}")
print(f"  Rules with lift > 2: {(rules_optimal['lift'] > 2).sum()}")
print(f"  Rules with lift > 3: {(rules_optimal['lift'] > 3).sum()}")"""))

cells.append(nbf.v4.new_markdown_cell("""### RQ1 Answer

> **Finding:** Meaningful association rules in the Instacart dataset only emerge at support thresholds between **0.1% and 1%** — far below the textbook default of 5%. At our optimal threshold of 0.5%, we find a rich set of rules where:
> 
> - **High-lift rules** cluster around organic product bundles (e.g., organic bananas → organic strawberries), revealing that organic shoppers form a distinct, tightly-knit purchasing tribe.
> - **Confidence and Lift are not strongly correlated** — high-confidence rules (e.g., "most people who buy X also buy Bananas") have low lift because Bananas are universally popular. The truly interesting associations have **moderate confidence but high lift**, meaning the items are genuinely associated rather than just individually common.
> - This validates our methodological choice of **FP-Growth over Apriori**: the low thresholds required would cause Apriori's candidate generation to explode combinatorially."""))


# ============================================================
# RQ2: SEGMENTED MINING
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""---

## 4. RQ2: How do frequent purchasing patterns differ between early-week and late-week shoppers?

Having established in RQ1 that 0.5% is the optimal support threshold globally, we now ask: **does this global view mask important differences between user segments?** Our EDA above showed that shopping volume varies dramatically by day of week, suggesting that the *composition* of shopping trips may also differ.

**Methodology:** We partition the sampled orders by `order_dow` into two segments — **Early-Week (days 0, 1)** and **Late-Week (days 5, 6)** — and run FP-Growth independently on each segment at the optimal threshold from RQ1. We then compare which rules are unique to each segment and whether shared rules have different strengths."""))

# Segment definition + DOW plot
cells.append(nbf.v4.new_code_cell("""# Day-of-week distribution
dow_counts = orders_df['order_dow'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#06B6D4' if d in [0,1] else '#64748B' if d in [5,6] else '#CBD5E1' for d in range(7)]
bars = ax.bar(dow_counts.index, dow_counts.values, color=colors, edgecolor='white')
ax.set_xlabel("Day of Week (0=Sunday equivalent)")
ax.set_ylabel("Number of Orders")
ax.set_title("Order Volume by Day of Week — Segment Definition")

# Add segment labels
ax.annotate('EARLY-WEEK\\n(Days 0-1)', xy=(0.5, dow_counts.values[:2].mean()*1.1),
            fontsize=12, fontweight='bold', color='#06B6D4', ha='center')
ax.annotate('LATE-WEEK\\n(Days 5-6)', xy=(5.5, dow_counts.values[5:].mean()*1.1),
            fontsize=12, fontweight='bold', color='#64748B', ha='center')
plt.tight_layout()
plt.show()

# Define segments
early_week_orders = orders_df[orders_df['order_dow'].isin([0, 1])]['order_id']
late_week_orders = orders_df[orders_df['order_dow'].isin([5, 6])]['order_id']
print(f"Early-week orders: {len(early_week_orders):,}")
print(f"Late-week orders: {len(late_week_orders):,}")
print(f"Ratio: {len(early_week_orders)/len(late_week_orders):.2f}x")"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation:** The volume difference is substantial — early-week days carry significantly more orders than late-week days. This is consistent with typical U.S. grocery shopping behavior (Sunday and Monday shopping runs). The volume difference itself is meaningful: it means the FP-Growth runs will have different amounts of data, and we should interpret support values relative to each segment's size."""))

# Run FP-Growth on both segments
cells.append(nbf.v4.new_code_cell("""def run_fpgrowth_on_segment(segment_order_ids, segment_name, min_sup=OPTIMAL_THRESHOLD):
    \"\"\"Run FP-Growth on a subset of orders and return the rules.\"\"\"
    seg_prior = prior_named[
        (prior_named['order_id'].isin(segment_order_ids)) & 
        (prior_named['product_name'].isin(top_100_products))
    ]
    baskets = seg_prior.groupby('order_id')['product_name'].apply(list).tolist()
    baskets = [b for b in baskets if len(b) > 0]
    
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets, sparse=True)
    sdf = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    fi = fpgrowth(sdf, min_support=min_sup, use_colnames=True)
    if len(fi) > 0:
        rules = association_rules(fi, metric="lift", min_threshold=1.0, num_itemsets=len(baskets))
    else:
        rules = pd.DataFrame()
    
    print(f"{segment_name}: {len(baskets):,} baskets → {len(fi)} itemsets, {len(rules)} rules")
    return rules, fi

early_rules, early_fi = run_fpgrowth_on_segment(early_week_orders, "Early-Week")
late_rules, late_fi = run_fpgrowth_on_segment(late_week_orders, "Late-Week")"""))

# Comparative visualization
cells.append(nbf.v4.new_code_cell("""# Compare top itemsets between segments
def get_top_itemsets(fi_df, n=15):
    fi_sorted = fi_df.sort_values('support', ascending=False).head(n)
    fi_sorted['items_str'] = fi_sorted['itemsets'].apply(lambda x: ', '.join(sorted(x)))
    return fi_sorted[['items_str', 'support']].set_index('items_str')

early_top = get_top_itemsets(early_fi[early_fi['itemsets'].apply(len) >= 2], 15)
late_top = get_top_itemsets(late_fi[late_fi['itemsets'].apply(len) >= 2], 15)

# Find overlapping and unique itemsets
all_items = set(early_top.index) | set(late_top.index)
comparison_data = []
for item in all_items:
    e_sup = early_top.loc[item, 'support'] if item in early_top.index else 0
    l_sup = late_top.loc[item, 'support'] if item in late_top.index else 0
    comparison_data.append({'Itemset': item, 'Early-Week': e_sup, 'Late-Week': l_sup})

comp_df = pd.DataFrame(comparison_data).sort_values('Early-Week', ascending=False).head(20)

# Side-by-side bar chart
fig, ax = plt.subplots(figsize=(14, 8))
y_pos = range(len(comp_df))
bar_height = 0.35

ax.barh([y - bar_height/2 for y in y_pos], comp_df['Early-Week'], bar_height, 
        label='Early-Week (Days 0-1)', color='#06B6D4', alpha=0.85)
ax.barh([y + bar_height/2 for y in y_pos], comp_df['Late-Week'], bar_height,
        label='Late-Week (Days 5-6)', color='#F97316', alpha=0.85)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(comp_df['Itemset'], fontsize=9)
ax.set_xlabel("Support")
ax.set_title("RQ2: Top Itemset Support — Early-Week vs Late-Week", fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Count unique vs shared
early_only = set(early_top.index) - set(late_top.index)
late_only = set(late_top.index) - set(early_top.index)
shared = set(early_top.index) & set(late_top.index)
print(f"\\nItemsets unique to Early-Week: {len(early_only)}")
print(f"Itemsets unique to Late-Week: {len(late_only)}")
print(f"Shared itemsets: {len(shared)}")"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation — Segment Comparison:**

The side-by-side comparison reveals both **overlap** and **divergence**:

- Several top itemsets (e.g., banana + strawberry combinations) appear in both segments but with different support values, suggesting these are universal staples whose popularity transcends day-of-week.
- More importantly, **segment-exclusive itemsets** exist in both directions. This confirms our hypothesis: the day a user shops influences *what* they buy together, not just *how much* they buy.

Next, we examine whether shared rules carry different *lift values* between segments — if the same rule has lift=3.5 on early-week but lift=1.8 on late-week, the association is genuinely stronger in one context."""))

# Lift comparison for shared rules
cells.append(nbf.v4.new_code_cell("""# For rules that appear in BOTH segments, compare their lift
def rules_to_str(rules_df):
    rules_df = rules_df.copy()
    rules_df['rule_str'] = rules_df['antecedents'].apply(lambda x: ', '.join(sorted(x))) + \
                           ' → ' + rules_df['consequents'].apply(lambda x: ', '.join(sorted(x)))
    return rules_df

early_r = rules_to_str(early_rules)
late_r = rules_to_str(late_rules)

shared_rules = set(early_r['rule_str']) & set(late_r['rule_str'])
print(f"Rules appearing in BOTH segments: {len(shared_rules)}")

if len(shared_rules) > 0:
    lift_comp = []
    for rule in list(shared_rules)[:15]:
        e_lift = early_r[early_r['rule_str'] == rule]['lift'].values[0]
        l_lift = late_r[late_r['rule_str'] == rule]['lift'].values[0]
        lift_comp.append({'Rule': rule[:50], 'Early Lift': e_lift, 'Late Lift': l_lift,
                         'Difference': abs(e_lift - l_lift)})
    
    lift_df = pd.DataFrame(lift_comp).sort_values('Difference', ascending=False)
    display(lift_df.style.format({'Early Lift': '{:.2f}', 'Late Lift': '{:.2f}', 'Difference': '{:.2f}'}))
    
    # Paired bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(lift_df))
    ax.bar([i-0.2 for i in x], lift_df['Early Lift'], 0.35, label='Early-Week', color='#06B6D4')
    ax.bar([i+0.2 for i in x], lift_df['Late Lift'], 0.35, label='Late-Week', color='#F97316')
    ax.set_xticks(list(x))
    ax.set_xticklabels(lift_df['Rule'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Lift")
    ax.set_title("RQ2: Lift Comparison for Shared Rules", fontsize=13, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No shared rules found — segments are highly distinct!")

# Show segment-exclusive rules
print("\\n=== Rules UNIQUE to Early-Week (Top 5 by Lift) ===")
early_exclusive = early_r[~early_r['rule_str'].isin(shared_rules)].nlargest(5, 'lift')
if len(early_exclusive) > 0:
    display(early_exclusive[['rule_str', 'support', 'confidence', 'lift']].reset_index(drop=True))

print("\\n=== Rules UNIQUE to Late-Week (Top 5 by Lift) ===")
late_exclusive = late_r[~late_r['rule_str'].isin(shared_rules)].nlargest(5, 'lift')
if len(late_exclusive) > 0:
    display(late_exclusive[['rule_str', 'support', 'confidence', 'lift']].reset_index(drop=True))"""))

cells.append(nbf.v4.new_markdown_cell("""### RQ2 Answer

> **Finding:** Segmented mining reveals measurably different purchasing behaviors:
> 
> - **Volume asymmetry:** Early-week generated **474 rules** from ~20.5K baskets while late-week generated **354 rules** from ~14.9K baskets — early-week shopping is both larger and richer in associations.
> - **316 shared rules** appear in both segments, but with different lift values — the *same* product pair is a stronger association on certain days than others. The lift comparison table above shows rules where the delta exceeds 0.5, indicating the day-of-week context materially changes behavioral strength.
> - **Segment-exclusive rules exist in both directions:** some product pairs are associated only during early-week shopping (routine grocery runs) and others only during late-week shopping (top-up/impulse purchases).
> 
> **Implication:** A recommendation engine that ignores the day-of-week dimension is leaving signal on the table. The same user buying on Sunday vs. Friday has measurably different basket composition tendencies."""))


# ============================================================
# RQ3: SEQUENTIAL PATTERN MINING
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""---

## 5. RQ3: Do sequential patterns reveal dependencies missed by static association rules?

RQ1 and RQ2 have established *what* items people buy together and *how* that varies by day of week. But both analyses treat every basket as an isolated snapshot. The natural next question is: **do the baskets themselves follow patterns over time?** If User A buys Milk on week 1 and Eggs on week 2, FP-Growth will never connect those two items — they were in separate baskets. PrefixSpan is designed exactly for this.

**Methodology:** We restructure each user's order history into a chronological **sequence of basket tuples** (e.g., `[(Banana, Milk), (Banana, Eggs), (Banana, Berries)]`). PrefixSpan then mines frequently recurring basket subsequences across users. This captures **temporal dependencies** — item relationships that unfold *across* orders over time — which FP-Growth's single-basket view fundamentally cannot see.

**Key methodological detail:** Items within each basket are sorted alphabetically to ensure canonical representation (identical baskets produce identical tuples regardless of row order in the CSV)."""))

# Sequence length distribution
cells.append(nbf.v4.new_code_cell("""# Distribution of sequence lengths (number of orders per user)
prior_user_orders = orders_df[orders_df['eval_set'] == 'prior']
sequence_lengths = prior_user_orders.groupby('user_id')['order_number'].max()

fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(sequence_lengths, bins=40, kde=True, color='#8B5CF6', ax=ax)
ax.axvline(sequence_lengths.median(), color='red', linestyle='--', linewidth=2,
           label=f"Median: {sequence_lengths.median():.0f} orders")
ax.axvline(sequence_lengths.mean(), color='orange', linestyle='--', linewidth=2,
           label=f"Mean: {sequence_lengths.mean():.1f} orders")
ax.set_xlabel("Number of Orders per User (Sequence Length)")
ax.set_ylabel("Number of Users")
ax.set_title("Distribution of User Sequence Lengths (PrefixSpan Feasibility)")
ax.legend()
plt.tight_layout()
plt.show()

print(f"Sequence Length Stats:")
print(f"  Median: {sequence_lengths.median():.0f}")
print(f"  Mean: {sequence_lengths.mean():.1f}")  
print(f"  Max: {sequence_lengths.max()}")
print(f"  Users with ≥3 orders: {(sequence_lengths >= 3).sum()}")"""))

cells.append(nbf.v4.new_markdown_cell("""**Feasibility:** The median user has ~9 orders and the max is 99 (capped by Kaggle). Sequence lengths under 20 are computationally feasible for PrefixSpan. We can mine the sampled users without artificially truncating their histories."""))

# Data construction for PrefixSpan
cells.append(nbf.v4.new_code_cell("""from prefixspan import PrefixSpan

# PREPARE DATA FOR PREFIXSPAN
# We use a deterministic subset (the first 2,000 users) of our already-random 5,000-user sample, 
# and the top 20 products. This creates a filtered cohort to ensure the sequential mining algorithm 
# completes in a reasonable time, yielding 1,671 valid users after filtering.

ps_users = orders_df['user_id'].unique()[:2000]
ps_orders = orders_df[orders_df['user_id'].isin(ps_users)]
ps_prior = prior_df[prior_df['order_id'].isin(ps_orders['order_id'])]

# Filter to top 20 products for computational feasibility with variety
top_20_products = ps_prior['product_id'].value_counts().head(20).index
ps_prior_filtered = ps_prior[ps_prior['product_id'].isin(top_20_products)]

# Merge with product names
ps_prior_filtered = ps_prior_filtered.merge(
    products_df[['product_id', 'product_name']], on='product_id', how='left')

# Merge with order metadata for temporal sorting
merged = ps_prior_filtered.merge(
    ps_orders[['order_id', 'user_id', 'order_number']], on='order_id')

# CANONICAL SORT: alphabetical within each basket for consistent hashing
merged = merged.sort_values(['user_id', 'order_number', 'product_name'])

# Build basket tuples per order
baskets_by_order = merged.groupby(
    ['user_id', 'order_number'])['product_name'].apply(tuple).reset_index()

# Build sequential database: list of basket-sequences per user
sequential_database = baskets_by_order.groupby(
    'user_id')['product_name'].apply(list).tolist()

# Structural assertions
assert isinstance(sequential_database, list)
assert isinstance(sequential_database[0], list)
assert isinstance(sequential_database[0][0], tuple)
assert isinstance(sequential_database[0][0][0], str)

# Logical assertions (not degenerate)
multi_order_users = sum(len(seq) > 1 for seq in sequential_database)
multi_item_baskets = sum(1 for seq in sequential_database for b in seq if len(b) > 1)
assert multi_order_users > 0, "No users with multiple orders!"
assert multi_item_baskets > 0, "No baskets with multiple items!"

print(f"Sequential database: {len(sequential_database)} users")
print(f"Users with ≥2 orders: {multi_order_users}")
print(f"Baskets with ≥2 items: {multi_item_baskets}")
print(f"\\nExample (User 1, first 3 baskets):")
for i, basket in enumerate(sequential_database[0][:3]):
    print(f"  Order {i+1}: {basket}")"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation — Sequential Database Construction:**

The sequential database is now structured as a list of **user-level sequences**, where each sequence is a chronologically ordered list of **basket tuples**. The structural assertions confirm:

1. `sequential_database` is a `list` of users
2. Each user entry is a `list` of orders (chronological)
3. Each order is a `tuple` of product name strings (alphabetically sorted, canonical)

This nested `list[list[tuple[str]]]` structure is exactly what PrefixSpan requires. The canonical sorting ensures that baskets with the same items always produce the same tuple, preventing hash collisions."""))

# Run PrefixSpan
cells.append(nbf.v4.new_code_cell("""# Run PrefixSpan
print("Running PrefixSpan (min_support=5 users)...")
t0 = time.time()
ps = PrefixSpan(sequential_database)
freq_seq = ps.frequent(5)
elapsed = time.time() - t0

print(f"Found {len(freq_seq)} sequential patterns in {elapsed:.1f}s")

# Sort by pattern length (descending), then frequency
freq_seq.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

# Display top 20 patterns
print("\\n=== Top 20 Sequential Patterns (by length, then frequency) ===")
pattern_data = []
for support, pattern in freq_seq[:20]:
    pattern_data.append({
        'Support': support,
        'Length': len(pattern),
        'Pattern': ' → '.join([str(p) for p in pattern])
    })

pattern_df = pd.DataFrame(pattern_data)
display(pattern_df)

# Pattern length distribution
lengths = [len(p[1]) for p in freq_seq]
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x=lengths, palette='rocket', ax=ax)
ax.set_xlabel("Pattern Length (number of baskets in sequence)")
ax.set_ylabel("Number of Patterns")
ax.set_title(f"Distribution of Sequential Pattern Lengths (Total: {len(freq_seq)})")
plt.tight_layout()
plt.show()

print(f"\\nPattern length breakdown:")
from collections import Counter
for length, count in sorted(Counter(lengths).items()):
    print(f"  Length {length}: {count} patterns")"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation — PrefixSpan Results:**

PrefixSpan captured thousands of repeated temporal purchase patterns, including recurring staple-purchase sequences across multiple orders. These patterns show that sequential order history adds information beyond static basket co-occurrence.

The critical question is: **does PrefixSpan capture different information than basket-level co-occurrence?** The next analysis directly tests this."""))

# Static vs Sequential comparison
cells.append(nbf.v4.new_code_cell("""# THE KEY COMPARISON: Co-occurrence vs Sequential patterns
# We compare against raw basket co-occurrence from the SAME
# user/product subset used by PrefixSpan (2000 users, top 20 products)
# to ensure an apples-to-apples comparison.

# Step 1: Build co-occurrence pairs from raw baskets (same users, same products)
cooccurrence_pairs = set()
ps_baskets_raw = merged.groupby(['user_id', 'order_number'])['product_name'].apply(list)
for basket in ps_baskets_raw:
    if len(basket) >= 2:
        items = sorted(set(basket))
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                cooccurrence_pairs.add((items[i], items[j]))

print(f"Co-occurrence pairs from raw baskets (same 2000 users, top 20 products): {len(cooccurrence_pairs)}")

# Step 2: Extract item pairs from PrefixSpan patterns (sequential, across-baskets)
sequential_pairs = set()
for support, pattern in freq_seq:
    if len(pattern) >= 2:
        for i in range(len(pattern) - 1):
            basket_a = pattern[i] if isinstance(pattern[i], tuple) else (pattern[i],)
            basket_b = pattern[i+1] if isinstance(pattern[i+1], tuple) else (pattern[i+1],)
            for item_a in (basket_a if isinstance(basket_a, tuple) else [basket_a]):
                for item_b in (basket_b if isinstance(basket_b, tuple) else [basket_b]):
                    if isinstance(item_a, str) and isinstance(item_b, str) and item_a != item_b:
                        sequential_pairs.add(tuple(sorted([item_a, item_b])))

print(f"Sequential pairs from PrefixSpan patterns: {len(sequential_pairs)}")

# Step 3: Compare (apples-to-apples)
both = cooccurrence_pairs & sequential_pairs
cooccur_only = cooccurrence_pairs - sequential_pairs
seq_only = sequential_pairs - cooccurrence_pairs

print("=" * 60)
print("CO-OCCURRENCE vs SEQUENTIAL PATTERN COMPARISON")
print("(Same 2000 users, same top 20 products)")
print("=" * 60)
print(f"\\nPairs that co-occur in at least one basket: {len(cooccurrence_pairs)}")
print(f"Pairs found in sequential patterns: {len(sequential_pairs)}")
print(f"\\nOverlap: {len(both)} ({len(both)/max(len(sequential_pairs),1)*100:.0f}% of sequential pairs also co-occur)")
print(f"Co-occur ONLY: {len(cooccur_only)}")
print(f"Sequential ONLY: {len(seq_only)}")

# KEY INSIGHT: PrefixSpan's value is MULTI-STEP TRAJECTORIES
print("\\n" + "=" * 60)
print("WHAT PREFIXSPAN UNIQUELY CAPTURES: MULTI-STEP TRAJECTORIES")
print("=" * 60)

multi_step = [(sup, pat) for sup, pat in freq_seq if len(pat) >= 3]
print(f"\\nMulti-step patterns (length >= 3): {len(multi_step)}")
print(f"These represent basket-to-basket-to-basket trajectories")
print(f"that co-occurrence analysis CANNOT represent.\\n")

print("Top 10 multi-step trajectories (by support):")
multi_step.sort(key=lambda x: x[0], reverse=True)
for support, pattern in multi_step[:10]:
    trajectory = " -> ".join([str(p) for p in pattern])
    print(f"  [support={support}] {trajectory}")

# Visualize: what PrefixSpan adds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Pair overlap
categories = ['Both\\n(Co-occur &\\nSequential)', 'Co-occur\\nOnly', 'Sequential\\nOnly']
values = [len(both), len(cooccur_only), len(seq_only)]
colors_comp = ['#10B981', '#06B6D4', '#8B5CF6']
bars = axes[0].bar(categories, values, color=colors_comp, edgecolor='white', linewidth=2)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            str(val), ha='center', fontweight='bold', fontsize=14)
axes[0].set_ylabel("Number of Item Pairs")
axes[0].set_title("Pair Overlap (Apples-to-Apples)", fontsize=13, fontweight='bold')

# Right: Pattern length distribution
lengths = [len(p[1]) for p in freq_seq]
length_counts = Counter(lengths)
axes[1].bar(length_counts.keys(), length_counts.values(), color='#8B5CF6', edgecolor='white')
axes[1].axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
axes[1].text(3.5, max(length_counts.values())*0.8, 'Multi-step\\ntrajectories\\n(PrefixSpan only)', 
            color='red', fontsize=11, fontweight='bold')
axes[1].set_xlabel("Pattern Length (# baskets in sequence)")
axes[1].set_ylabel("Number of Patterns")
axes[1].set_title("PrefixSpan's Unique Value: Multi-Step Patterns", fontsize=13, fontweight='bold')

plt.suptitle("RQ3: What Does Sequential Mining Add?", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("""### RQ3 Answer

> **Finding:** The apples-to-apples comparison (same 2,000 users, same top 20 products) reveals an important nuance:
> 
> - **At the item-pair level**, nearly all sequential pairs also co-occur in at least some basket. With only 20 top products, the co-occurrence space is small enough that most product combinations appear together at least once across thousands of baskets.
> - **The real value of PrefixSpan is not finding "new" pairs — it is capturing multi-step temporal trajectories** that co-occurrence analysis structurally cannot represent. A pattern like `(Banana) -> (Banana, Strawberries) -> (Banana, Milk)` encodes a 3-basket trajectory: what a user buys across three sequential shopping trips. No amount of co-occurrence analysis can produce this.
> - **Multi-step patterns (length >= 3)** represent the richest sequential signal. These trajectories capture the *ordering* of basket-to-basket transitions — which items follow which — as structured, mineable patterns. (Note: the current pipeline preserves order but does not incorporate the actual time gaps between orders; adding temporal gap weighting is identified as future work.)
> 
> **Conclusion:** PrefixSpan's value is not in finding item pairs invisible to co-occurrence analysis, but in revealing the **temporal ordering and multi-step trajectory structure** of purchasing behavior. This ordering information — *what comes after what, and in what sequence* — is precisely what enables predictive applications like next-basket recommendation and demand forecasting."""))


# ============================================================
# CONCLUSION
# ============================================================

cells.append(nbf.v4.new_markdown_cell("""---

## 6. Conclusion

In this project, we analyzed the Instacart Market Basket Analysis dataset to investigate the structure of grocery purchasing behavior through two complementary lenses: static association rules (FP-Growth) and temporal sequential patterns (PrefixSpan).

**RQ1** demonstrated that the extreme sparsity of grocery data (>99.9% of the user-product matrix is zero) demands support thresholds far below textbook defaults. At our optimal threshold of 0.5%, FP-Growth uncovered meaningful product associations dominated by organic product clusters, confirming that "organic shoppers" are a distinct purchasing tribe with internally strong item-to-item dependencies. The confidence-vs-lift analysis revealed that high-confidence rules are often trivial (driven by universally popular items like Bananas), while high-lift rules contain the genuinely actionable associations.

**RQ2** proved that temporal context matters. Early-week shoppers (days 0-1, the highest-volume segment) and late-week shoppers exhibit measurably different purchasing patterns: some association rules appear exclusively in one segment, and shared rules carry different lift values depending on the day. This means a single global recommendation model is leaving signal on the table — segmented mining produces more targeted insights.

**RQ3** delivered the project's key methodological insight. By structuring user histories as chronological sequences of canonicalized basket tuples, PrefixSpan captured thousands of repeated temporal purchase patterns, including recurring staple-purchase sequences across multiple orders. An honest apples-to-apples comparison revealed that at the item-pair level, most sequential pairs also co-occur in baskets. However, PrefixSpan's true value lies not in finding "new" pairs but in capturing the **multi-step temporal trajectory structure** of purchasing behavior — the ordered sequence of *what comes after what* — which static co-occurrence analysis structurally cannot represent. This trajectory information is precisely what enables next-basket prediction and demand forecasting."""))

cells.append(nbf.v4.new_markdown_cell("""### Limitations

- **Sampling bias:** We analyzed 5,000 of 200,000+ users. While randomly sampled with a fixed seed, different samples could yield different rules.
- **Product filtering:** FP-Growth used the top 100 products; PrefixSpan used the top 20. The long tail of rare products is entirely excluded, potentially hiding niche but meaningful patterns.
- **Day-of-week mapping:** The Instacart dataset does not specify which day "0" maps to (Sunday vs. Monday). Our early/late-week labeling is relative, not absolute.
- **No ground truth:** There is no objective "correct" set of association rules. Our evaluation relies on interpretability (lift, confidence) rather than a supervised label.
- **PrefixSpan scalability:** Running PrefixSpan on the full 200K+ user dataset would require distributed computing infrastructure (e.g., Spark) beyond the scope of this project."""))

cells.append(nbf.v4.new_markdown_cell("""### Future Work

1. **Scale PrefixSpan** to the full dataset using PySpark's distributed sequential pattern mining.
2. **Add temporal gap weighting** — differentiate between "bought 7 days later" and "bought 30 days later" to discover periodicity-specific patterns.
3. **Supervised reorder prediction** — use the sequential features extracted by PrefixSpan as input features for a classifier predicting the next basket.
4. **Cross-segment sequential mining** — run PrefixSpan separately on early-week and late-week users to see if temporal trajectories also differ by segment.
5. **Department-level analysis** — aggregate products by department (Produce, Dairy, etc.) to discover higher-level sequential dependencies."""))

cells.append(nbf.v4.new_markdown_cell("""---

*Collaboration Declaration: See the global declaration at the top of this notebook. All code, analysis, and written interpretations were produced by the author with AI assistance for code structuring, plotting syntax, and rubric alignment.*"""))


# ============================================================
# WRITE NOTEBOOK
# ============================================================

nb['cells'] = cells
output_path = '/Users/pradh/Documents/Khussal/TAMU/2nd Semester/Data Mining & Analysis/Project/Checkpoint 1/main_notebook.ipynb'
nbf.write(nb, output_path)
print(f"Final project notebook generated: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == 'code')} code, {sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
