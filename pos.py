# --- Simple rule-based POS Tagger ---


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --- 1️⃣ Input sentence ---
sentence = "I am learning natural language processing"

# --- 2️⃣ Tokenize the sentence into words ---
words = nltk.word_tokenize(sentence)

# --- 3️⃣ Perform POS tagging ---
pos_tags = nltk.pos_tag(words)

# --- 4️⃣ Display results ---
print("🧠 Words:", words)
print("🏷️ POS Tags:", pos_tags)


def simple_pos_tagger(sentence):
    words = sentence.split()
    pos_tags = []

    # --- small lexicon ---
    lexicon = {
        "i": "PRON", "you": "PRON", "he": "PRON", "she": "PRON", "it": "PRON",
        "we": "PRON", "they": "PRON",
        "is": "VERB", "am": "VERB", "are": "VERB", "was": "VERB", "were": "VERB",
        "eat": "VERB", "eats": "VERB", "run": "VERB", "runs": "VERB",
        "apple": "NOUN", "banana": "NOUN", "fruit": "NOUN",
        "happy": "ADJ", "sad": "ADJ", "big": "ADJ", "small": "ADJ",
        "quickly": "ADV", "slowly": "ADV"
    }

    # --- simple rule-based tagging ---
    for word in words:
        w = word.lower()

        if w in lexicon:
            tag = lexicon[w]

        elif w.endswith("ly"):
            tag = "ADV"
        elif w.endswith("ing"):
            tag = "VERB"
        elif w.endswith("ed"):
            tag = "VERB"
        elif w.endswith("ness") or w.endswith("tion"):
            tag = "NOUN"
        elif w.endswith("ous") or w.endswith("ful") or w.endswith("able"):
            tag = "ADJ"
        elif w.istitle():
            tag = "NOUN"   # assume proper noun
        else:
            tag = "NOUN"   # default

        pos_tags.append((word, tag))

    return pos_tags


# --- test ---
sentence = "She is running quickly towards the big apple"
tags = simple_pos_tagger(sentence)

print("🧠 Sentence:", sentence)
print("🏷️ POS Tags:", tags)
