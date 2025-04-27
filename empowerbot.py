# Install required libraries (only run once)
# !pip install sentence-transformers faiss-cpu transformers streamlit

# Import libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import streamlit as st

# Step 1: Build Knowledge Base
legal_texts = [
    # your list of legal texts here (same as before)...
    "Is dowry illegal in India? -> Yes, under the Dowry Prohibition Act, 1961, giving, taking, or demanding dowry is illegal and punishable by up to 3 years of imprisonment and fines.",
    "What are a woman's rights after marriage? -> A woman has rights to residence, maintenance, dignity, and protection against domestic violence under Indian law.",
    # ... rest of the entries
]

# Step 2: Encode Knowledge Base
@st.cache_resource
def load_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(legal_texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index

model, index = load_embeddings()

# Step 3: Load Lightweight LLM
@st.cache_resource
def load_generator():
    generator = pipeline('text-generation', model='tiiuae/falcon-7b-instruct', max_new_tokens=100)
    return generator

generator = load_generator()

# Step 4: Retrieval + Generation
def empowerbot_answer(user_query):
    query_embedding = model.encode([user_query])
    D, I = index.search(np.array(query_embedding), k=1)
    matched_text = legal_texts[I[0][0]]
    prompt = f"You are an expert legal assistant for women. Based on the following law: {matched_text} \n\nAnswer the user's question: {user_query}"
    output = generator(prompt)[0]['generated_text']
    return output

# Step 5: Feedback Loop
def feedback_loop(user_query, correct_answer):
    with open('feedback_data.txt', 'a') as f:
        f.write(f"Query: {user_query}\nCorrect Answer: {correct_answer}\n\n")
    return "Thank you for your feedback!"

# --- Streamlit UI ---
st.title("EmpowerBot - Women's Legal Assistant")
st.write("Ask EmpowerBot about women's rights and get reliable legal information.")

# User query input
user_input = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_input.strip() != "":
        answer = empowerbot_answer(user_input)
        st.text_area("EmpowerBot's Answer", value=answer, height=200)

# Feedback Section
st.subheader("Provide Feedback")
user_query_feedback = st.text_input("Your original query (for feedback):", key="feedback_query")
correct_answer_feedback = st.text_area("Correct Answer you'd like to suggest:", key="feedback_answer")

if st.button("Submit Feedback"):
    if user_query_feedback.strip() != "" and correct_answer_feedback.strip() != "":
        message = feedback_loop(user_query_feedback, correct_answer_feedback)
        st.success(message)
