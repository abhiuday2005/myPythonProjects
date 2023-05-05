import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


with open('text.txt', 'r') as f:
    text = f.read()


scores = sia.polarity_scores(text)


print("Sentiment scores:")
print(f"Positive: {scores['pos']:.2f}")
print(f"Negative: {scores['neg']:.2f}")
print(f"Neutral: {scores['neu']:.2f}")
print(f"Compound: {scores['compound']:.2f}")
