# Solution for Slicing and Indexing Mini Practice Task
import pandas as pd
import numpy as np

# Create a DataFrame with data about different products
products_data = {
    'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing', 'Home', 'Electronics', 'Books'],
    'price': [120.50, 45.99, 89.99, 34.50, 120.00, 199.99, 399.99, 12.99],
    'stock_quantity': [25, 50, 5, 30, 8, 15, 3, 100]
}

products_df = pd.DataFrame(products_data)

# Display the original DataFrame
print("Original Products DataFrame:")
print(products_df)

# Use boolean indexing to find products that are low in stock (less than 10)
low_stock_products = products_df[products_df['stock_quantity'] < 10]
print("\nProducts with Low Stock (less than 10):")
print(low_stock_products)

# Use boolean indexing to find expensive products (more than $100)
expensive_products = products_df[products_df['price'] > 100]
print("\nExpensive Products (more than $100):")
print(expensive_products)

# Use boolean indexing to find products that are either low in stock OR expensive
low_stock_or_expensive = products_df[(products_df['stock_quantity'] < 10) | (products_df['price'] > 100)]
print("\nProducts that are either Low in Stock OR Expensive:")
print(low_stock_or_expensive)

# Use boolean indexing to find products that are both low in stock AND expensive
low_stock_and_expensive = products_df[(products_df['stock_quantity'] < 10) & (products_df['price'] > 100)]
print("\nProducts that are both Low in Stock AND Expensive:")
print(low_stock_and_expensive)

# Set the product_id as the index
products_indexed = products_df.set_index('product_id')
print("\nProducts DataFrame with product_id as index:")
print(products_indexed)

# Access products by their index
print("\nProduct P003 details:")
print(products_indexed.loc['P003'])

# Create a multi-level index using category and product_id
products_multi_index = products_df.set_index(['category', 'product_id'])
print("\nProducts DataFrame with multi-level index:")
print(products_multi_index)

# Access products by category
print("\nAll Electronics products:")
print(products_multi_index.loc['Electronics'])

# Print a summary
print("\nSummary of Inventory Status:")
for category, group in products_df.groupby('category'):
    low_stock_count = len(group[group['stock_quantity'] < 10])
    expensive_count = len(group[group['price'] > 100])
    total_count = len(group)
    
    print(f"{category}: {total_count} products, {low_stock_count} low stock, {expensive_count} expensive")

# Example output:
# Original Products DataFrame:
#   product_id    category   price  stock_quantity
# 0       P001  Electronics  120.50             25
# 1       P002     Clothing   45.99             50
# 2       P003  Electronics   89.99              5
# 3       P004        Home   34.50             30
# 4       P005     Clothing  120.00              8
# 5       P006        Home  199.99             15
# 6       P007  Electronics  399.99              3
# 7       P008       Books   12.99            100
# 
# Products with Low Stock (less than 10):
#   product_id    category   price  stock_quantity
# 2       P003  Electronics   89.99              5
# 4       P005     Clothing  120.00              8
# 6       P007  Electronics  399.99              3
# 
# Expensive Products (more than $100):
#   product_id    category   price  stock_quantity
# 0       P001  Electronics  120.50             25
# 4       P005     Clothing  120.00              8
# 5       P006        Home  199.99             15
# 6       P007  Electronics  399.99              3
# 
# Products that are either Low in Stock OR Expensive:
#   product_id    category   price  stock_quantity
# 0       P001  Electronics  120.50             25
# 2       P003  Electronics   89.99              5
# 4       P005     Clothing  120.00              8
# 5       P006        Home  199.99             15
# 6       P007  Electronics  399.99              3
# 
# Products that are both Low in Stock AND Expensive:
#   product_id    category   price  stock_quantity
# 4       P005     Clothing  120.00              8
# 6       P007  Electronics  399.99              3