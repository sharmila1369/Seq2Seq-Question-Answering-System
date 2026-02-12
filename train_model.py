import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding


df = pd.read_csv("qa_dataset.csv")

questions = df["question"].values
answers = ["<start> " + a + " <end>" for a in df["answer"].values]


tokenizer_q = Tokenizer()
tokenizer_q.fit_on_texts(questions)
seq_q = tokenizer_q.texts_to_sequences(questions)

tokenizer_a = Tokenizer()
tokenizer_a.fit_on_texts(answers)
seq_a = tokenizer_a.texts_to_sequences(answers)


with open("model/tokenizer_q.pkl", "wb") as f:
    pickle.dump(tokenizer_q, f)

with open("model/tokenizer_a.pkl", "wb") as f:
    pickle.dump(tokenizer_a, f)


max_len_q = 15
max_len_a = 15

encoder_input = pad_sequences(seq_q, maxlen=max_len_q, padding="post")
decoder_input = pad_sequences(seq_a, maxlen=max_len_a, padding="post")


decoder_target = np.zeros_like(decoder_input)
decoder_target[:, :-1] = decoder_input[:, 1:]
decoder_target = np.expand_dims(decoder_target, -1)


vocab_q = len(tokenizer_q.word_index) + 1
vocab_a = len(tokenizer_a.word_index) + 1
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_len_q,))
enc_emb = Embedding(vocab_q, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_len_a,))
dec_emb = Embedding(vocab_a, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_a, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

print(model.summary())


model.fit([encoder_input, decoder_input], decoder_target,
          batch_size=8, epochs=200)

# Save model (LOCAL MACHINE)
model.save("model/seq2seq_qna.h5")
print("Model trained and saved!")
