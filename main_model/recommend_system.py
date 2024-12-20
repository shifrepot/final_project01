import sqlite3
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import re
import json
import os
from openai import OpenAI

# OpenAI API 설정
print("[INFO] OpenAI API 설정 중...")
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
print("[INFO] OpenAI API 설정 완료!")

# 모델 로드
print("[INFO] 모델 로드 중...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
analyzer = pipeline("text-classification", model="newih/finetuning-sentiment-model-10000-samples")
print("[INFO] 모델 로드 완료!")

def cosine_similarity(vec1, vec2):
    """
    두 벡터 간의 코사인 유사도 계산
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def extract_movie_title(user_input):
    """
    LLM을 사용하여 입력에서 영화 제목 추출
    """
    print("[INFO] 영화 제목 추출 중...")
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"사용자의 입력에서 영화 제목을 찾아주세요.\n입력: \"{user_input}\""}
        ]
    )
    movie_title = completion.choices[0].message.content.strip()
    print(f"[INFO] 추출된 영화 제목: {movie_title}")
    return movie_title

def search_movie_in_db(user_input, model, conn):
    """
    사용자 입력을 기반으로 DB에서 영화 검색 (임베딩 기반 유사도)
    """
    print("[INFO] DB에서 영화 검색 중...")
    user_embedding = model.encode(user_input)

    cursor = conn.cursor()
    cursor.execute("SELECT title, genres, overview, release_year, embedding FROM movies")
    results = cursor.fetchall()

    similarities = [
        (row[0], row[1], row[2], row[3], cosine_similarity(user_embedding, np.array(json.loads(row[4]))))
        for row in results
    ]
    similarities = sorted(similarities, key=lambda x: x[4], reverse=True)
    print(f"[INFO] 검색된 영화 수: {len(similarities)}")
    return similarities[:5]

def query_llm_for_movies(user_input):
    """
    LLM을 사용하여 영화 추천
    """
    print("[INFO] LLM에서 추천 영화 생성 중...")
    prompt = f"""
    사용자 입력: {user_input}
    추천할 영화를 생성하세요. 제목, 간단한 설명, 장르를 포함해주세요.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    print("[INFO] LLM에서 영화 추천 완료!")
    return response.choices[0].message.content

def recommend_movies(user_input):
    """
    영화 추천 시스템의 전체 흐름
    """
    print("[INFO] 영화 추천 작업 시작!")

    # Step 1: 영화 제목 추출
    movie_title = extract_movie_title(user_input)

    # Step 2: 감정 분석
    print("[INFO] 감정 분석 중...")
    sentiment_result = analyzer(user_input)
    sentiment, sentiment_score = sentiment_result[0]['label'], sentiment_result[0]['score']
    print(f"[INFO] 감정 분석 결과: {sentiment} (점수: {sentiment_score:.2f})")

    # Step 3: DB 검색
    print("[INFO] DB 검색 중...")
    conn = sqlite3.connect("movies_with_embeddings.db")
    results = search_movie_in_db(user_input, model, conn)
    conn.close()

    # Step 4: 결과 생성
    if results:
        print("[INFO] DB 검색 결과 기반 추천 생성 중...")
        recommendations = [f"Title: {r[0]}, Genres: {r[1]}, Year: {r[3]}, Similarity: {r[4]:.2f}" for r in results]
        response = f"당신은 '{movie_title}'에 대해 {sentiment} ({sentiment_score:.2f}) 감정을 갖고 있군요.\n추천 영화:\n" + "\n".join(recommendations)
        print("[INFO] DB 검색 결과를 사용하여 추천 생성 완료!")
    else:
        print("[INFO] DB에서 관련 영화가 없어 LLM을 호출합니다...")
        llm_results = query_llm_for_movies(user_input)
        response = f"당신은 '{movie_title}'에 대해 {sentiment} ({sentiment_score:.2f}) 감정을 갖고 있군요.\nLLM 추천:\n{llm_results}"

    print("[INFO] 영화 추천 작업 완료!")
    return response

if __name__ == "__main__":
    user_input = input("영화에 대해 자유롭게 입력해주세요:\n")
    print(recommend_movies(user_input))
