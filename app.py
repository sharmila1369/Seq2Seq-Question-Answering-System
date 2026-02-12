from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


model = load_model("model/seq2seq_qna.h5")

with open("model/tokenizer_q.pkl", "rb") as f:
    tokenizer_q = pickle.load(f)

with open("model/tokenizer_a.pkl", "rb") as f:
    tokenizer_a = pickle.load(f)


max_len_q = 15
max_len_a = 15

start_idx = tokenizer_a.word_index.get("<start>", 1)
end_idx = tokenizer_a.word_index.get("<end>", 2)

def generate_answer(question):
    seq = tokenizer_q.texts_to_sequences([question])
    padded_q = pad_sequences(seq, maxlen=max_len_q, padding="post")

    target_seq = np.array([[start_idx]])
    decoded_sentence = ""

    for _ in range(max_len_a):
        padded_target = pad_sequences(target_seq, maxlen=max_len_a, padding="post")

        prediction = model.predict([padded_q, padded_target], verbose=0)
        next_token = np.argmax(prediction[0, target_seq.shape[1] - 1, :])

        if next_token == end_idx:
            break

        word = tokenizer_a.index_word.get(next_token, "")
        if word:
            decoded_sentence += word + " "

        target_seq = np.append(target_seq, [[next_token]], axis=1)

    return decoded_sentence.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = generate_answer(question)

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
