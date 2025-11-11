import numpy as np

# --- 1️⃣ Training data (tiny example) ---
# Each sentence is a list of (word, tag) pairs
data = [
    [("I", "PRON"), ("eat", "VERB"), ("apple", "NOUN")],
    [("You", "PRON"), ("eat", "VERB"), ("banana", "NOUN")],
    [("She", "PRON"), ("likes", "VERB"), ("fruit", "NOUN")]
]

# --- 2️⃣ Build vocabulary, tag set, and count probabilities ---
words = set()
tags = set()
for sent in data:
    for word, tag in sent:
        words.add(word)
        tags.add(tag)

# Initialize counts
transition_counts = {}  # P(tag_t | tag_{t-1})
emission_counts = {}    # P(word | tag)
tag_counts = {}

for sent in data:
    prev_tag = "<START>"
    for word, tag in sent:
        # transition
        transition_counts.setdefault(prev_tag, {})
        transition_counts[prev_tag][tag] = transition_counts[prev_tag].get(tag, 0) + 1
        
        # emission
        emission_counts.setdefault(tag, {})
        emission_counts[tag][word] = emission_counts[tag].get(word, 0) + 1
        
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        prev_tag = tag

# --- 3️⃣ Convert counts to probabilities ---
def normalize(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

transition_probs = {t: normalize(d) for t, d in transition_counts.items()}
emission_probs = {t: normalize(d) for t, d in emission_counts.items()}

tags = list(tags)

# --- 4️⃣ Viterbi Algorithm ---
def viterbi(sentence, tags, transition_probs, emission_probs):
    V = [{}]
    path = {}

    # Initialization step
    for tag in tags:
        V[0][tag] = transition_probs["<START>"].get(tag, 0.0001) * emission_probs[tag].get(sentence[0], 0.0001)
        path[tag] = [tag]

    # Recursion
    for t in range(1, len(sentence)):
        V.append({})
        newpath = {}
        for tag in tags:
            (prob, prev_tag) = max(
                [(V[t-1][ptag] *
                  transition_probs.get(ptag, {}).get(tag, 0.0001) *
                  emission_probs[tag].get(sentence[t], 0.0001), ptag)
                 for ptag in tags]
            )
            V[t][tag] = prob
            newpath[tag] = path[prev_tag] + [tag]
        path = newpath

    # Termination
    n = len(sentence) - 1
    (prob, final_tag) = max([(V[n][tag], tag) for tag in tags])
    return path[final_tag]

# --- 5️⃣ Test sentence ---
sentence = ["I", "like", "banana"]
tags_pred = viterbi(sentence, tags, transition_probs, emission_probs)
print("🧠 Sentence:", sentence)
print("🏷️ Predicted POS tags:", tags_pred)
