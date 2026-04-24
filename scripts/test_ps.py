import pandas as pd
from prefixspan import PrefixSpan

orders = pd.read_csv("kaggleInstacart/orders.csv")
prior = pd.read_csv("kaggleInstacart/order_products__prior.csv")

# 500 users
micro_users = orders['user_id'].unique()[:500]
micro_orders = orders[orders['user_id'].isin(micro_users)]
micro_prior = prior[prior['order_id'].isin(micro_orders['order_id'])]

# Only keep top 5 most popular products across the dataset
top_products = micro_prior['product_id'].value_counts().head(5).index
micro_prior = micro_prior[micro_prior['product_id'].isin(top_products)]

merged = micro_prior.merge(micro_orders[['order_id', 'user_id', 'order_number']], on='order_id')
merged = merged.sort_values(['user_id', 'order_number'])

baskets_by_order = merged.groupby(['user_id', 'order_number'])['product_id'].apply(tuple).reset_index()
seq_db = baskets_by_order.groupby('user_id')['product_id'].apply(list).tolist()

ps = PrefixSpan(seq_db)
print(f"Freq (10 users): {len(ps.frequent(10))}")
print(f"Freq (5 users): {len(ps.frequent(5))}")
print(f"Freq (2 users): {len(ps.frequent(2))}")
