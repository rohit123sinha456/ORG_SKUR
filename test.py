import pandas as pd
from db import get_connection
import ast
import numpy as np
conn = get_connection()
# Read embeddings and any other relevant columns
query = "SELECT * FROM items"
df = pd.read_sql(query, conn)
x = np.array(ast.literal_eval(df['filtered_itemdesc_embedding'].iloc[0]), dtype=float)
print(x.shape)

# np.array(ast.literal_eval(emb), dtype=float)