import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 키 가져오기

def query_llm_for_movies(user_input):
    prompt = f"""
    사용자 입력: {user_input}
    추천할 영화를 생성하세요. 영화 제목, 간단한 설명, 장르를 포함해주세요.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
