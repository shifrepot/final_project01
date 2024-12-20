from transformers import pipeline

# 감정 분석 모델 로드
analyzer = pipeline("text-classification", model="newih/finetuning-sentiment-model-10000-samples")

def analyze_sentiment(review):
    """
    리뷰에 대한 감정 분석 수행
    """
    result = analyzer(review)
    return result[0]['label'], result[0]['score']
