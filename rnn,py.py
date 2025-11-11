import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# --- 1️⃣ Sample dataset ---
texts = [
    "I love this movie",
    "This film is amazing",
    "I hate this movie",
    "This film is terrible",
    "Absolutely wonderful experience",
    "Worst movie ever"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = positive, 0 = negative

# --- 2️⃣ Tokenize and pad ---
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
maxlen = 6  # maximum sentence length

X = pad_sequences(sequences, maxlen=maxlen)
y = tf.constant(labels)

# --- 3️⃣ Build RNN model ---
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=50, input_length=maxlen),
    SimpleRNN(32, activation='tanh'),
    Dense(1, activation='sigmoid')  # binary output
])

# --- 4️⃣ Compile and train ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, verbose=1)

# --- 5️⃣ Test prediction ---
test_texts = ["I love this film", "This movie was awful"]
test_seq = tokenizer.texts_to_sequences(test_texts)
test_pad = pad_sequences(test_seq, maxlen=maxlen)
preds = model.predict(test_pad)

for t, p in zip(test_texts, preds):
    print(f"{t} --> {'Positive' if p > 0.5 else 'Negative'} ({p[0]:.2f})")
