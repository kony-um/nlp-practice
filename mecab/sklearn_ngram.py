from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenize

texts = [
    '東京から大阪に行く',
    '大阪から東京に行く'
]

vectorizer = CountVectorizer(tokenizer=tokenize, ngram_range=(2,2))
vectorizer.fit(texts)
bow = vectorizer.transform(texts)
