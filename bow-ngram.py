# Sentiment Classifier using Bag of Words and N-gram models

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "I love this movie, it was fantastic and enjoyable",
    "This film is terrible and boring",
    "Absolutely wonderful experience, I liked it a lot",
    "I hate this movie, it was awful",
    "The story was good but the acting was bad",
    "Amazing direction and great acting",
    "Worst movie ever made",
    "I really enjoyed the performance",
    "Not good, very disappointing",
    "Excellent and thrilling film"
]

labels = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# ---- Bag of Words Model ----
print("=== Bag of Words Model ===")

# Convert text to feature vectors (BoW)
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Train Naive Bayes classifier
bow_model = MultinomialNB()
bow_model.fit(X_train_bow, y_train)

# Predict and evaluate
y_pred_bow = bow_model.predict(X_test_bow)
print("Accuracy (BoW):", round(accuracy_score(y_test, y_pred_bow), 2))


# ---- N-gram Model (Bigrams + Trigrams) ----
print("\n=== N-gram Model (Bigrams + Trigrams) ===")

# Convert text to feature vectors using N-grams
ngram_vectorizer = CountVectorizer(ngram_range=(1, 3))  # includes unigrams, bigrams, trigrams
X_train_ng = ngram_vectorizer.fit_transform(X_train)
X_test_ng = ngram_vectorizer.transform(X_test)

# Train Naive Bayes classifier
ng_model = MultinomialNB()
ng_model.fit(X_train_ng, y_train)

# Predict and evaluate
y_pred_ng = ng_model.predict(X_test_ng)
print("Accuracy (N-gram):", round(accuracy_score(y_test, y_pred_ng), 2))
