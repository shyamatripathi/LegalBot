# EmpowerBot: Legal Rights Chatbot for Women

EmpowerBot is an NLP-powered chatbot designed to make legal information more accessible—especially for women seeking clarity about their rights under Indian law. Using semantic search, embeddings, and lightweight LLM-based response generation, EmpowerBot provides simplified and reliable legal guidance through an intuitive web interface.

---

## 🌟 Features

* **Retrieval-Based Question Answering (RAG)** using Sentence Transformers + FAISS
* **Curated knowledge base** covering 30+ legal topics related to women’s rights
* **Semantic similarity search** for accurate legal information retrieval
* **LLM-powered answer generation** with a helpful and empathetic tone
* **Feedback loop integration** for improving responses
* **Gradio Web UI** for seamless user interaction

---

##  How It Works

1. User enters a query related to women’s legal rights.
2. The query is converted into embeddings using **Sentence Transformers**.
3. **FAISS** retrieves the closest matching legal text.
4. A lightweight LLM generates a simplified and supportive response.
5. Optional feedback helps improve future answers.

---

##  Tech Stack

* Python
* Sentence Transformers
* FAISS (Vector Indexing & Search)
* Transformers / LLM
* Gradio
* NumPy

---

## 📂 Project Structure

```
├── app.py              # Main chatbot + RAG logic
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .devcontainer/      # Dev container configuration (VS Code)
```

---

## ▶️ Running the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The Gradio interface will start locally and provide a browser link.

---

##  Example Queries

* “What are my rights if I face domestic violence?”
* “How do I file a workplace harassment complaint?”
* “Is dowry legal in India?”
* “What are the grounds for divorce?”

---

## 📌 Disclaimer

EmpowerBot provides simplified legal information based on publicly available laws. It **does not replace professional legal counsel**.

---

## 🤝 Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request to enhance the knowledge base, improve retrieval accuracy, or refine the UI.
