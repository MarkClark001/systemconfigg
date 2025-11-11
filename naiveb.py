# Naive Bayes Text Classifier from scratch (very simple version)

import math
import string

# ---- 1. Sample training data ----
dataset = [
    ("I love this movie", "pos"),
    ("This film is amazing", "pos"),
    ("I hate this movie", "neg"),
    ("This film is terrible", "neg"),
    ("What a great experience", "pos"),
    ("Worst movie ever", "neg"),
]

# ---- 2. Preprocessing ----
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# ---- 3. Build vocabulary and word counts ----
word_counts = {}
class_counts = {}
total_docs = len(dataset)

for text, label in dataset:
    words = preprocess(text)
    class_counts[label] = class_counts.get(label, 0) + 1
    if label not in word_counts:
        word_counts[label] = {}
    for w in words:
        word_counts[label][w] = word_counts[label].get(w, 0) + 1

# ---- 4. Calculate priors ----
priors = {label: count / total_docs for label, count in class_counts.items()}

# ---- 5. Prediction function ----
def predict(text):
    words = preprocess(text)
    scores = {}

    for label in class_counts.keys():
        # Start with log prior
        scores[label] = math.log(priors[label])

        # Total word count in class
        total_words = sum(word_counts[label].values())
        vocab_size = len(set([w for d in word_counts.values() for w in d]))

        for w in words:
            # Apply Laplace smoothing
            word_likelihood = (word_counts[label].get(w, 0) + 1) / (total_words + vocab_size)
            scores[label] += math.log(word_likelihood)

    # Return label with highest probability
    return max(scores, key=scores.get)

# ---- 6. Test the classifier ----
test_texts = [
    "I love this film",
    "This movie was awful",
    "Great and amazing experience",
    "Terrible film and bad acting"
]

for t in test_texts:
    print(f"{t} --> {predict(t)}")
