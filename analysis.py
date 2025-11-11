# Simple Type-Token Analysis (word level)

import string

# Sample text
text = """Natural language processing enables computers 
to understand and generate human language efficiently."""

# Preprocessing: remove punctuation and lowercase
text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# Tokenize (split into words)
tokens = text.split()

# Count total tokens and unique types
num_tokens = len(tokens)
num_types = len(set(tokens))

# Compute Type-Token Ratio (TTR)
ttr = num_types / num_tokens

# Display results
print("Tokens (total words):", num_tokens)
print("Types (unique words):", num_types)
print("Type-Token Ratio (TTR):", round(ttr, 3))


# Simple Type–Token Analysis at the Syllable Level

import re
import string

def count_syllables(word):
    """
    Simple heuristic-based syllable counter.
    Counts vowel groups as syllables (approximation).
    """
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)  # keep only letters
    vowels = "aeiou"
    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups)
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)

def get_syllables(word):
    """
    Split a word into syllable-like chunks (simple approximation).
    """
    return re.findall(r'[^aeiou]*[aeiouy]+', word.lower())

# Sample text
text = """Natural language processing enables computers
to understand and generate human language efficiently."""

# Preprocess: remove punctuation and lowercase
text = text.translate(str.maketrans('', '', string.punctuation)).lower()

# Split into words
words = text.split()

# Extract syllables from each word
syllables = []
for w in words:
    syllables.extend(get_syllables(w))

# Type–Token counts
num_tokens = len(syllables)
num_types = len(set(syllables))
ttr = num_types / num_tokens

# Display results
print("Total syllable tokens:", num_tokens)
print("Unique syllable types:", num_types)
print("Type–Token Ratio (TTR):", round(ttr, 3))
print("\nSyllables:", syllables)
