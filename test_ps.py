import pandas as pd
from prefixspan import PrefixSpan

orders = pd.read_csv("kaggleInstacart/orders.csv")
prior = pd.read_csv("kaggleInstacart/order_products__prior.csv")

# 500 users
micro_users = orders['user_id'].unique()[:500]
micro_orders = orders[orders['user_id'].isin(micro_users)]
micro_prior = prior[prior['order_id'].isin(micro_orders['order_id'])]

# Only keep top 10 most popular products across the dataset to increase overlap chance
top_products = micro_prior['product_id'].value_counts().head(20).index
micro_prior = micro_prior[micro_prior['product_id'].isin(top_products)]

# Group into lists (since prefixspan treats tuples as atomic, let's use list of individual string items for simple sequence of top items)
# E.g. sequence of top items bought over time
merged = micro_prior.merge(micro_orders[['order_id', 'user_id', 'order_number']], on='order_id')
merged = merged.sort_values(['user_id', 'order_number'])

# Sequence of individual product IDs bought
seq_db = merged.groupby('user_id')['product_id'].apply(list).tolist()

ps = PrefixSpan(seq_db)
print(f"Sequences: {len(seq_db)}")
print(f"Freq: {len(ps.frequent(30))}") # occurs in 30 user histories
