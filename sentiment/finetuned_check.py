from transformers import pipeline
 
sentiment_model = pipeline(model="newih/finetuning-sentiment-model-10000-samples")
input_texts = ["The movie had good acting but the plot was predictable."]

#1,0 -> 1: positive, 0: negative

# 감성 분석 수행
results = sentiment_model(input_texts)

# 결과 출력
for text, result in zip(input_texts, results):
    label = result['label']  # 예측 레이블
    score = result['score']  # 해당 레이블의 신뢰도 점수
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {label} ({score:.2f})")
    print("-" * 30)
