# --- Install if needed ---
# pip install nltk

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# --- 1️⃣ Input sentence ---
sentence = "The quick brown fox jumps over the lazy dog"

# --- 2️⃣ Tokenize and POS-tag ---
words = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(words)

print("POS Tags:", pos_tags)

# --- 3️⃣ Define a chunk grammar ---
# NP = Noun Phrase (optional determiner + adjectives + noun)
# VP = Verb Phrase (verb + optional noun phrase)
grammar = r"""
  NP: {<DT>?<JJ>*<NN.*>}       # e.g. 'the lazy dog'
  VP: {<VB.*><NP|PP|RB>*}      # e.g. 'jumps over the dog'
"""

# --- 4️⃣ Create a chunk parser ---
chunk_parser = nltk.RegexpParser(grammar)

# --- 5️⃣ Apply shallow parsing ---
tree = chunk_parser.parse(pos_tags)

# --- 6️⃣ Display results ---
print("\nShallow Parse Tree:")
print(tree)

# Optional: visualize the tree
tree.draw()
