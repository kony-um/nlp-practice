from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenize

texts = [
    '私は私のことを好きなあなたが好き',
    '私はラーメンが好きです',
    '富士山は日本一高い山です',
]

vectorizer = CountVectorizer(tokenizer=tokenize)
vectorizer.fit(texts)
bow = vectorizer.transform(texts)