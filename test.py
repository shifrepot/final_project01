import sqlite3
import pandas as pd

# SQLite 연결
conn = sqlite3.connect("movies_with_embeddings.db")

# 데이터 확인
query = "SELECT * FROM movies LIMIT 5"
df = pd.read_sql(query, conn)

print("DB에서 로드된 데이터 샘플:")
print(df.head())

conn.close()
