# Simple Porter Stemmer (basic implementation from scratch)

def simple_porter_stemmer(word):
    """
    A simplified version of the Porter stemming algorithm.
    This version only demonstrates core steps conceptually.
    """

    # Step 1a: Remove common plurals and -ed/-ing endings
    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-3] + 'i'
    elif word.endswith('ss'):
        pass
    elif word.endswith('s'):
        word = word[:-1]

    # Step 1b: Remove -ed or -ing
    if word.endswith('eed'):
        word = word[:-1]
    elif word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('ing'):
        word = word[:-3]

    # Step 2: Replace 'ational' -> 'ate', 'tional' -> 'tion'
    if word.endswith('ational'):
        word = word[:-7] + 'ate'
    elif word.endswith('tional'):
        word = word[:-6] + 'tion'

    # Step 3: Remove a final 'e' if present
    if word.endswith('e'):
        word = word[:-1]

    return word


# Example usage
words = ["caresses", "ponies", "caressed", "running", "conditional", "rational", "baking"]
stems = [simple_porter_stemmer(w) for w in words]

for w, s in zip(words, stems):
    print(f"{w} --> {s}")
