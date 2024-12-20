import sqlite3
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json

# tqdm 설정
tqdm.pandas()

# 모델 로드 (GPU 사용 설정)
print("모델 로드 중...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')  # GPU 사용
print("모델 로드 완료!")

# 데이터 로드
print("데이터 로드 중...")
path_tmdb = "/root/own_cinema/TMDB_all_movies.csv"
df = pd.read_csv(path_tmdb)
print(f"데이터 로드 완료! 총 {len(df)}개의 영화 데이터")

# 데이터 전처리
print("데이터 전처리 중...")
columns_to_drop = [
    'id', 'imdb_id', 'vote_count', 'revenue', 'budget', 'original_title',
    'production_companies', 'production_countries',
    'director_of_photography', 'producers', 'music_composer',
    'imdb_votes', 'poster_path', 'imdb_rating', 'spoken_languages',
]
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned = df_cleaned[df_cleaned['status'] == 'Released']
df_cleaned['release_year'] = pd.to_datetime(df_cleaned['release_date'], errors='coerce').dt.year.astype('Int64')
df_cleaned = df_cleaned.drop(columns=['release_date', 'status'])
df_cleaned = df_cleaned.dropna(subset=['title'])
popularity_threshold = df_cleaned['popularity'].quantile(0.4)
vote_average_threshold = df_cleaned['vote_average'].quantile(0.4)
df_cleaned = df_cleaned[
    (df_cleaned['popularity'] > popularity_threshold) & 
    (df_cleaned['vote_average'] > vote_average_threshold)
]
df_cleaned = df_cleaned.drop(columns=['popularity', 'vote_average'])

print(f"데이터 전처리 완료! 총 {len(df_cleaned)}개의 데이터 남음.")

# 결합 텍스트 생성
print("결합 텍스트 생성 중...")
df_cleaned['combined_text'] = df_cleaned['title'] + " " + df_cleaned['genres'] + " " + df_cleaned['overview'].fillna("")
df_cleaned['combined_text'] = df_cleaned['combined_text'].fillna("").astype(str)

# 임베딩 생성 (GPU 사용)
print("임베딩 생성 중...")
df_cleaned['embedding'] = df_cleaned['combined_text'].progress_apply(lambda x: model.encode(x).tolist())
df_cleaned['embedding'] = df_cleaned['embedding'].apply(json.dumps)
print("임베딩 생성 완료!")

# SQLite 저장
print("SQLite 저장 중...")
conn = sqlite3.connect("movies_with_embeddings.db")
df_cleaned[['title', 'genres', 'overview', 'release_year', 'embedding']].to_sql(
    "movies", conn, if_exists="replace", index=False
)
conn.close()

print("DB 구성 완료!")
