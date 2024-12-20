Final_Project01: RAG 기반 영화 추천 시스템
=============

프로젝트 개요
-------------
이 프로젝트는 RAG(Retrieval-Augmented Generation) 기반의 영화 추천 시스템입니다.
사용자의 자연어 입력을 분석하여 감정을 파악하고, 데이터베이스(DB)와 LLM(GPT API)을 결합해 사용자의 취향에 맞는 영화를 추천합니다.
특히 감정 분석 모델과 유사도 검색 알고리즘, 그리고 GPT-4를 활용한 생성형 추천을 결합하여 강력한 사용자 경험을 제공합니다.

프로젝트 목표
-------------
사용자의 입력(영화 리뷰 및 의견)을 기반으로 취향과 감정에 맞는 영화를 추천합니다.
최소한의 자원(GPU, 메모리)을 사용하면서도 RAG 아키텍처를 활용해 효율적으로 동작합니다.
사용자 친화적인 챗봇 인터페이스를 제공하여 자연스러운 대화 흐름으로 결과를 제공합니다.

데이터셋
-------------
이 프로젝트에서는 다음과 같은 데이터셋을 사용했습니다:

###### MovieLens 32M Dataset

영화 평점, 태그 데이터를 포함한 대규모 안정적인 벤치마크 데이터셋.
총 3,200만 개의 평점, 200만 개의 태그, 87,585편의 영화를 포함.
데이터 구성:
ratings.csv: 영화 평점 데이터 (0.5 ~ 5.0).
tags.csv: 사용자가 작성한 영화에 대한 간단한 태그.
movies.csv: 영화 제목과 장르 데이터.
links.csv: TMDB, IMDb의 영화 ID를 매핑.
The Ultimate 1Million Movies Dataset

영화 제목, 개봉 정보, 평점, 인기 지수, 관객 평점, 제작사 등 포괄적인 정보를 포함.
TMDB와 IMDb 데이터를 결합하여 제작.

###### IMDb Dataset (from datasets 라이브러리)

영화 리뷰 텍스트 및 긍정/부정 레이블 포함.
감정 분석 모델 파인튜닝에 사용.
크기: 50,000개의 영화 리뷰 (25,000개의 훈련 데이터와 25,000개의 테스트 데이터).

아키텍처
-------------

###### 1. 전체 흐름
데이터 전처리 및 임베딩 생성:

영화 데이터를 전처리(title, genres, overview)한 후, SentenceTransformer로 임베딩 생성.
결과를 SQLite 데이터베이스에 저장.
감정 분석: 사용자의 입력 텍스트를 기반으로 긍정/부정을 분석.

#####DB 검색:사용자 입력 임베딩과 DB의 영화 임베딩 간 코사인 유사도를 계산.
가장 유사도가 높은 영화를 반환.

#####LLM 호출: DB에서 적절한 결과를 찾지 못한 경우 gpt-4o API를 호출해 새로운 영화를 생성.
결과 출력: 감정 분석 결과, DB 검색 결과 또는 gpt-4o 생성 결과를 자연어로 출력.
###### 2. 구현 세부사항
##### 2.1 데이터 전처리 및 임베딩 생성
#### 파일: db_setup.py
#### 기능:
불필요한 열 제거, 결합 텍스트(title, genres, overview) 생성.
SentenceTransformer로 텍스트 임베딩 생성.
SQLite 데이터베이스(movies_with_embeddings.db)에 저장.

##### 2.2 감정 분석
#### 파일: sentiment_analyzer.py
#### 모델: newih/finetuning-sentiment-model-10000-samples
#### 기능:
사용자 입력 텍스트의 감정을 긍정/부정으로 분류.
transformers 라이브러리의 pipeline 활용.

##### 2.3 영화 검색
#### 파일: recommend_system.py
#### 기능:
사용자 입력 텍스트를 임베딩화.
DB의 영화 임베딩과 코사인 유사도 계산.
가장 유사도가 높은 영화 5개 반환.

##### 2.4 GPT-4 기반 영화 추천
#### 파일: llm_query.py
#### 기능:
사용자의 입력을 GPT-4에 전달하여 관련 영화 추천 생성.
감정 분석 모델 파인튜닝

### 1. 파인튜닝 과정
데이터셋: IMDb Dataset (50,000개의 영화 리뷰)
모델: distilbert-base-uncased
훈련 데이터: 10,000개의 훈련 데이터와 1,000개의 테스트 데이터.
프레임워크: Hugging Face transformers.
### 2. 훈련 코드
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 데이터셋 로드 및 축소
imdb = load_dataset("imdb")
small_train_dataset = imdb["train"].shuffle(seed=42).select(range(10000))
small_test_dataset = imdb["test"].shuffle(seed=42).select(range(1000))

# 토크나이저 및 전처리
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# 모델 정의
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./finetuned_sentiment_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# 모델 훈련
trainer.train()
trainer.save_model("./finetuned_sentiment_model")
### 3. 모델 성능
정확도(Accuracy): 89.3%
F1 Score: 88.7%
pip install -r requirements.txt

export OPENAI_API_KEY="your_openai_api_key"
데이터베이스 생성:
python db_setup.py -> movies_with_embeddings.db 로 저장
* * *
영화 추천 시스템 실행:
python recommend_system.py
* * *
입력:
"I absolutely loved Titanic. It made me so emotional!"
* * *
출력:
[INFO] 영화 추천 작업 시작!
[INFO] 영화 제목 추출 중...
[INFO] 추출된 영화 제목: Titanic
[INFO] 감정 분석 중...
[INFO] 감정 분석 결과: positive (0.92)
[INFO] DB 검색 중...
[INFO] DB에서 영화 검색 중...
[INFO] 검색된 영화 수: 5
[INFO] DB 검색 결과 기반 추천 생성 중...
[INFO] DB 검색 결과를 사용하여 추천 생성 완료!
당신은 'Titanic'에 대해 positive (0.92) 감정을 갖고 있군요.
추천 영화:
Title: Titanic 2, Genres: Drama, Romance, Year: 2010, Similarity: 0.87
Title: Avatar, Genres: Sci-Fi, Adventure, Year: 2009, Similarity: 0.82
* * *
파일 구조
project/
├── db_setup.py                 # DB 구성 및 임베딩 생성 코드
├── sentiment_analyzer.py       # 감정 분석 코드
├── llm_query.py                # GPT-4 호출 코드
├── recommend_system.py         # 전체 시스템 통합 및 실행 코드
├── movies_with_embeddings.db   # SQLite 데이터베이스
├── requirements.txt            # 필요한 패키지 목록
└── README.md                   # 프로젝트 설명 파일
