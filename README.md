# final_project01
final project for 기계학습과 코딩
# final_project01
final project for 기계학습과 코딩

# own_cinema
Unraveling the perfect movie match for users with a touch of RAG-powered intelligence. Explore personalized recommendations and detailed movie insights!

Goal : 본 프로젝트의 목표는 사용자의 입력을 바탕으로 사용자의 취향과 상황에 맞는 영화를 추천하는 것입니다.

Dataset : 본 프로젝트에 사용된 데이터셋은 다음과 같습니다.

1) MovieLens 32M dataset
 영화 평점과 태그 데이터를 포함한 stable benchmark dataset
 2023년 10월에 수집되어 2024년 5월에 공개되었으며, 총 3,200만 개의 평점과 200만 개의 태그가 200,948명의 사용자에 의해 87,585편의 영화에 적용
 links.csv, movies.csv, ratings.csv, tags.csv 로 구성
ratings: 0.5~5.0 사이의 영화 rating 데이터
tags : 사용자가 입력한 영화에 대한 짧은 단어/구 메타데이터
movies : 영화 제목, 장르
links : tdmb, imdb, movie ID
 2) The Ultimate 1Million Movies Dataset (TMDB + IMDb)
Kaggle 에 upload 된 dataset, 
영화 제목 및 개봉 정보, 장르 및 키워드, 평점 및 인기 지수, 관객 평점, 인기 점수, 투표 수 데이터, 제작사, 제작 국가, 예산, 출연/제작진,
음악 감독 등 포괄적 정보가 담긴 데이터셋
3) datasets library 의 idmb dataset
영화 리뷰 분석 및 자연어 처리(NLP) 연구에 자주 사용되는 데이터셋 중 하나
영화 리뷰 텍스트와 해당 레이블(긍정/부정)을 포함하여 감정 분석(Sentiment Analysis) 모델 훈련 및 평가에 사용
크기: 약 50,000개의 영화 리뷰로 구성.(25,000개의 훈련 데이터.25,000개의 테스트 데이터.)

프로젝트 과정
"발전된 영화 추천 시스템" 이 목표!
추천시스템이란?
AI/ML algorithm을 이용하여 빅데이터 기반으로 사용자가 관심을 가질 제품이나 상품(본 프로젝트에서는 영화)을 추천해 주는 것

보통 추천 시스템의 유형은 협업 필터링(collaborative filtering), 콘텐츠 기반 필터링(content-based filtering), hybrid filtering, 그리고 context filtering으로 구성
콘텐츠 기반 필터링은 사용자 간 선호 유사도를 기반으로 추천하는 것 -> 비슷한 영화를 좋아한, 나와 비슷한 유저를 찾아 추천해주는 것
이것을 원래 movie lens dataset의 tag와 ratings, 그리고 userid, movie id 묶어서 대신하려고 했음. 왜냐하면 나는 netflix나 amazon 같이 user의 개인화된 정보를 얻을 수 없었기 때문. 

원래는 RAG을 계획하였었음 -> movie lens dataset에서는 user ID, move ID, user가 부여한 tags와 user가 부여한 ratings가 존재하는데, 이를 이용해 GNN이나 neural network 에 넣고 추천 시스템에서 collaborative filtering처럼 활용하고자 하였음. 그리고 영화 메타데이터를 vector database를 활용하여 업로드 하여 content based filtering으로 활용하고자 하였음. 
그러나,실제로 저 data 를 실제 통계적으로 분석한 결과, 원래 가설이었던 "user 들은 특정 tag 에 대해 ratings를 높게 주어 다른 dataset이 없더라도 
