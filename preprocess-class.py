# Simple Text Classifier with Preprocessing

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download required NLTK data
nltk.download('stopwords')

# Sample dataset
texts = [
    "I love programming in Python!",
    "Python is an amazing language.",
    "I dislike bugs in the code.",
    "Debugging code can be very frustrating."
]
labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative sentiment

# Preprocessing function
def preprocess(text):
    # a) Remove punctuations and symbols
    text = text.translate(str.maketrans('', '', string.punctuation))
    # b) Convert to lowercase
    text = text.lower()
    # c) Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    # d) Apply stemming
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in words]
    return " ".join(stemmed)

# Apply preprocessing
processed_texts = [preprocess(t) for t in texts]

# e) Convert words into vectors (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train a simple classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Test and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
