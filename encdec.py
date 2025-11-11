import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

# --- 1️⃣ Small toy dataset (English → Spanish) ---
english_texts = [
    "hello",
    "how are you",
    "good morning",
    "thank you",
    "i love you",
    "see you later"
]

spanish_texts = [
    "inicio hola fin",
    "inicio como estas fin",
    "inicio buenos dias fin",
    "inicio gracias fin",
    "inicio te amo fin",
    "inicio hasta luego fin"
]
# we add 'inicio' (start) and 'fin' (end) tokens to help the decoder learn when to start and stop

# --- 2️⃣ Tokenize and pad ---
eng_tokenizer = Tokenizer()
spa_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(english_texts)
spa_tokenizer.fit_on_texts(spanish_texts)

eng_seq = eng_tokenizer.texts_to_sequences(english_texts)
spa_seq = spa_tokenizer.texts_to_sequences(spanish_texts)

max_eng_len = max(len(s) for s in eng_seq)
max_spa_len = max(len(s) for s in spa_seq)

eng_vocab = len(eng_tokenizer.word_index) + 1
spa_vocab = len(spa_tokenizer.word_index) + 1

X = pad_sequences(eng_seq, maxlen=max_eng_len, padding='post')
y = pad_sequences(spa_seq, maxlen=max_spa_len, padding='post')

# Decoder input and target (shifted by one)
decoder_input_data = y[:, :-1]
decoder_target_data = y[:, 1:]
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# --- 3️⃣ Define Encoder–Decoder Model ---
latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(max_eng_len,))
enc_emb = Embedding(eng_vocab, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_spa_len - 1,))
dec_emb = Embedding(spa_vocab, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(spa_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([X, decoder_input_data], decoder_target_data, batch_size=2, epochs=200, verbose=0)
print("✅ Training complete!")

# --- 4️⃣ Define inference models ---

# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(spa_vocab, latent_dim)(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_states2 = [state_h2, state_c2]

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

# --- 5️⃣ Translation function ---
reverse_spa_index = {v: k for k, v in spa_tokenizer.word_index.items()}

def translate_sentence(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    
    states_value = encoder_model.predict(seq)
    
    target_seq = np.array([[spa_tokenizer.word_index['inicio']]])
    decoded_sentence = ''
    
    for _ in range(max_spa_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_spa_index.get(sampled_token_index, '')
        
        if sampled_word == 'fin' or sampled_word == '':
            break
        
        decoded_sentence += sampled_word + ' '
        
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    
    return decoded_sentence.strip()

# --- 6️⃣ Try translations ---
print("\n🔤 Example translations:")
for test in ["hello", "good morning", "i love you"]:
    print(f"{test} → {translate_sentence(test)}")
