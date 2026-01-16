# University IT Assistant – RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) chatbot for IT and computer science Q&A, powered by Google Gemini (GenAI), sentence-transformers, and FAISS. Built with Flask for easy deployment and extensibility.

---

## 🚀 Features

- **RAG Architecture**: Combines semantic search (sentence-transformers + FAISS) with generative answers from Gemini (Google GenAI).
- **Custom Knowledge Base**: Uses `intents.json` for domain-specific Q&A (IT, computer science, etc.).
- **Fast & Accurate Retrieval**: Embeds and indexes knowledge base for efficient, relevant context retrieval.
- **Professional, Original Answers**: Gemini generates clear, non-verbatim responses using only retrieved context.
- **Easy API**: Simple `/ask` endpoint for integration with web or other clients.
- **Web UI**: Modern, responsive chat interface (`index.html`).
- **Debug Endpoint**: `/debug` for health and config info.

---

## 🛠️ Setup & Installation

1. **Clone the repository**
	```bash
	git clone <repo-url>
	cd chatbot
	```

2. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```

3. **Set up your environment**
	- Create a `.env` file with your Gemini/Google API key:
	  ```env
	  GEMINI_API_KEY=your_google_gemini_api_key
	  ```
	- (Optional) Adjust model/config in `.env` as needed (see `app.py` for options).

4. **Prepare your knowledge base**
	- Edit `intents.json` to add or update Q&A pairs.

5. **Run the server**
	```bash
	python app.py
	```
	- Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 How It Works

1. **User asks a question** (via web UI or API).
2. **Semantic search** retrieves the most relevant Q&A pairs from the knowledge base using sentence-transformers and FAISS.
3. **Gemini (GenAI)** receives the question and retrieved context, and generates a professional, original answer (never copies verbatim).
4. **Response** is returned to the user, with context and similarity scores available for debugging.

---


## 📊 Datasets Used

- **intents.json**: The main dataset for the chatbot, containing structured Q&A pairs (intents, patterns, and responses) used for both retrieval and training.
- **it_domain_questions.csv**: An additional dataset with IT domain questions, used to enrich, validate, and experiment with the knowledge base and model performance.

These datasets form the foundation of the chatbot's knowledge and were essential in the creation and evaluation of the entire project.

---

## 📒 Notebooks

The `notebooks/` directory contains Jupyter notebooks used for:
- Data exploration and preprocessing
- Experimenting with NLP techniques (TF-IDF, Word2Vec, GloVe, KNN, SVM, LSTM, seq2seq, attention, etc.)
- Training and evaluating models
- Prototyping and validating the RAG pipeline before production

These notebooks document the research and development process, and can be used to reproduce or extend the experiments.

---

## 📦 Project Structure

- `app.py` – Main Flask backend (RAG logic, API endpoints)
- `intents.json` – Knowledge base (editable Q&A pairs)
- `index.html` – Web chat UI
- `requirements.txt` – Python dependencies
- `notebooks/` – Experiments, model training, and analysis
- `static/` – Static assets (if needed)
- `docs/` – Documentation

---

## 🔗 API Usage

### POST `/ask`

Request:
```json
{
  "question": "What is a binary search tree?"
}
```

Response:
```json
{
  "question": "What is a binary search tree?",
  "answer": "A binary search tree (BST) is ...",
  "used_gemini": true,
  "contexts_found": [
	 {"tag": "bst", "similarity": 0.98},
	 ...
  ]
}
```

---

## 📝 Customization

- **Add new Q&A**: Edit `intents.json` (see format inside the file).
- **Change embedding/model**: Adjust `EMBED_MODEL` or `GENIE_MODEL` in `.env` or `app.py`.
- **Tune retrieval**: Change `TOP_K` or `SIMILARITY_THRESHOLD` in `.env` or `app.py`.

---

## 📚 Requirements

- Python 3.9+
- See `requirements.txt` for all Python dependencies

---

## 🤖 Example Questions

- "Explain data abstraction."
- "What is a syntax error?"
- "Describe decision tree."
- "What is logistic regression?"

---

## 🛡️ License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author & Credits

- Developed by Ayoub hannachi

---
