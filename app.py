"""
app.py
RAG + Gemini (Google GenAI) example using:
 - sentence-transformers (all-MiniLM-L6-v2)
 - faiss (IndexFlatIP with L2-normalized vectors for cosine similarity)
 - google-genai SDK (client.models.generate_content)
Requires: Python 3.9+
"""

import os
import json
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# embeddings / FAISS
from sentence_transformers import SentenceTransformer
import faiss

# Google GenAI SDK
from google import genai

# ----------------- Configuration -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment or .env")

# Model selection (adjust if needed)
DEFAULT_GENIE_MODEL = os.getenv("GENIE_MODEL", "gemini-3-flash-preview")

# Knowledge base & embedding settings
INTENTS_PATH = os.getenv("INTENTS_PATH", "intents.json")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "300"))

# ----------------- Initialize SDKs -----------------
client = genai.Client(api_key=GOOGLE_API_KEY)

print("Loading sentence-transformers embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# ----------------- Load & Build Knowledge Base -----------------
def load_intents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("intents", [])

def build_knowledge_base(intents):
    kb = []
    for intent in intents:
        tag = intent.get("tag", "")
        patterns = intent.get("patterns", [])
        responses = intent.get("responses", [])
        for p in patterns:
            resp = responses[0] if responses else ""
            kb.append({
                "tag": tag,
                "pattern": p,
                "response": resp,
                "text": f"Q: {p}\nA: {resp}"
            })
    return kb

INTENTS = load_intents(INTENTS_PATH)
KNOWLEDGE_BASE = build_knowledge_base(INTENTS)

# ----------------- Build FAISS index -----------------
if len(KNOWLEDGE_BASE) == 0:
    print("⚠️ Knowledge base is empty. Make sure 'intents.json' exists.")

texts = [entry["text"] for entry in KNOWLEDGE_BASE]
if texts:
    embs = embedder.encode(texts, show_progress_bar=False)
    embs = np.array(embs).astype("float32")
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    print(f"FAISS index built. {index.ntotal} vectors indexed.")
else:
    index = None

# ----------------- Retrieval -----------------
def retrieve_relevant_context(question, top_k=TOP_K):
    if index is None:
        return []
    q_emb = embedder.encode([question]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) >= SIMILARITY_THRESHOLD:
            results.append({
                "context": KNOWLEDGE_BASE[idx],
                "score": float(score)
            })
    return results

# ----------------- Generate Answer with Gemini -----------------
def generate_answer_with_gemini(question, contexts):
    if not contexts:
        return ("I don't have a relevant answer in my knowledge base for that question.", False)

    ref_texts = [
        f"Reference {i+1} (tag: {c['context']['tag']}, score: {c['score']:.3f}):\n{c['context']['text']}"
        for i, c in enumerate(contexts[:2])
    ]
    references_block = "\n\n".join(ref_texts)

    user_prompt = f"""You are a knowledgeable university-level IT assistant. 
You have access to the following references extracted from a knowledge base. 
Your task is to generate a **clear, original, professional answer** to the question using only the information in the references. 
**Do not copy the references verbatim.** Summarize, rephrase, or combine the information if needed. 
If the answer is not present in the references, respond: "I don't know based on the provided references."

References:
{references_block}

QUESTION:
{question}

ANSWER:"""


    try:
        response = client.models.generate_content(
            model=DEFAULT_GENIE_MODEL,
            contents=user_prompt,
            
        )
        text = response.text.strip() if getattr(response, "text", None) else str(response)
        return (text, True)
    except Exception as e:
        print("Gemini API call failed:", repr(e))
        fallback = contexts[0]["context"]["response"]
        fallback_msg = f"(Gemini error fallback) {fallback}" if fallback else "I don't know."
        return (fallback_msg, False)

# ----------------- Flask App -----------------
app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/debug")
def debug():
    return jsonify({
        "status": "online",
        "knowledge_entries": len(KNOWLEDGE_BASE),
        "faiss_index_size": index.ntotal if index else 0,
        "embedding_model": EMBED_MODEL,
        "genie_model": DEFAULT_GENIE_MODEL
    })

@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(force=True)
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    contexts = retrieve_relevant_context(question)
    answer, used_gemini = generate_answer_with_gemini(question, contexts)

    return jsonify({
        "question": question,
        "answer": answer,
        "used_gemini": used_gemini,
        "contexts_found": [
            {"tag": c["context"]["tag"], "similarity": c["score"]} for c in contexts
        ]
    })

if __name__ == "__main__":
    print("🎓 RAG Flask server running...")
    app.run(host="0.0.0.0", port=5000, debug=True)
