# Install required libraries
!pip install sentence-transformers faiss-cpu gradio transformers

# Import libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import gradio as gr

# Step 1: Build Knowledge Base from Legal Texts
legal_texts = [
    # Dowry and Marriage
    "Is dowry illegal in India? -> Yes, under the Dowry Prohibition Act, 1961, giving, taking, or demanding dowry is illegal and punishable by up to 3 years of imprisonment and fines.",
    "What are a woman's rights after marriage? -> A woman has rights to residence, maintenance, dignity, and protection against domestic violence under Indian law.",

    # Divorce
    "How can a woman file for divorce? -> A woman can file for divorce on grounds such as cruelty, adultery, desertion, conversion, mental disorder, and mutual consent under the Hindu Marriage Act, 1955.",
    "Can a divorced woman claim maintenance? -> Yes, under Section 125 of the Criminal Procedure Code (CrPC), a divorced woman can claim maintenance if she cannot maintain herself.",

    # Domestic Violence
    "What protections does a woman have under the Domestic Violence Act? -> The Protection of Women from Domestic Violence Act, 2005 grants women the right to protection orders, residence, monetary relief, and child custody.",
    "What is domestic violence under Indian law? -> Domestic violence includes physical, emotional, sexual, verbal, and economic abuse against a woman within the family.",

    # Workplace Rights
    "What rights do women have at the workplace? -> Under the POSH Act, 2013, women have the right to a safe working environment and can file complaints about sexual harassment at the workplace.",
    "How can a woman report sexual harassment at work? -> She can file a complaint with the Internal Complaints Committee (ICC) established under the POSH Act, 2013.",

    # Property and Inheritance
    "Can daughters inherit ancestral property? -> Yes, under the Hindu Succession (Amendment) Act, 2005, daughters have equal rights to ancestral property as sons.",
    "What happens to a woman's property after marriage? -> A woman's property remains solely hers after marriage unless she chooses to transfer or gift it voluntarily.",

    # Maternity Benefits
    "What are a woman's rights during pregnancy at work? -> Under the Maternity Benefit Act, 1961, a woman is entitled to 26 weeks of paid maternity leave for the first two children.",
    "Is it legal to fire a woman during maternity leave? -> No, dismissing a woman during maternity leave is illegal under the Maternity Benefit Act.",

    # Police and Legal Assistance
    "How can a woman file an FIR for harassment? -> A woman can directly approach a police station to file an FIR. As per Supreme Court guidelines, no woman can be denied lodging an FIR.",
    "Can a woman file an online police complaint? -> Yes, many Indian states offer online FIR and grievance redressal portals where women can file complaints remotely.",

    # Sexual Offenses and Safety
    "What are the laws against sexual assault in India? -> Sections 354, 354A, 354B, 354C, and 354D of IPC deal with outraging a woman's modesty, harassment, stalking, and voyeurism.",
    "Is marital rape recognized in Indian law? -> As of now, marital rape is not fully criminalized under Indian law except under certain circumstances like child marriage or separation cases.",
    
    # Child Marriage
    "Is child marriage illegal in India? -> Yes, under the Prohibition of Child Marriage Act, 2006, marriage below 18 years (for girls) and 21 years (for boys) is illegal and punishable.",

    # Human Trafficking and Forced Labour
    "What protections exist against human trafficking? -> The Immoral Traffic (Prevention) Act, 1956 criminalizes trafficking of women and girls for prostitution.",
    "Can women working as domestic helpers claim protection? -> Yes, domestic workers are entitled to protection under labor laws and state-level welfare boards.",

    # Cyber Crimes
    "What are a woman's rights against cyberbullying? -> A woman facing online abuse, threats, or revenge porn can report under the Information Technology Act, 2000 and IPC Section 66A, 67, 67A.",
    "How can a woman report cybercrime? -> She can file a complaint at the nearest Cyber Crime Cell or use the online portal: cybercrime.gov.in.",

    # Financial Rights
    "Are women entitled to joint property after marriage? -> In case of divorce, women can claim maintenance and residential rights but automatic rights over husband's property depend on various factors.",
    "Do women have banking rights without husband/family permission? -> Yes, adult women have the full legal right to operate independent bank accounts and investments.",

    # Widow Rights
    "What are a widow's rights to husband's property? -> A widow has a rightful share in her deceased husband's property under the Hindu Succession Act, 1956.",

    # Child Custody
    "What are a woman's rights regarding child custody after divorce? -> Custody decisions are based on the child's welfare. Generally, mothers are granted custody for minor children unless proven unfit.",

    # Protection Against Acid Attacks
    "What legal protections exist against acid attacks? -> Acid attacks are punishable under IPC Sections 326A and 326B with imprisonment up to 10 years or life, along with fines.",
    "Can acid attack victims get compensation? -> Yes, acid attack victims are entitled to free medical treatment and compensation under Supreme Court rulings and state schemes.",

    # Reproductive Rights
    "Can a woman legally terminate a pregnancy? -> Under the Medical Termination of Pregnancy (MTP) Act, 1971, a woman can legally terminate a pregnancy up to 24 weeks under certain conditions.",

# Miscellaneous
    "Do women have the right to equal pay? -> Yes, under the Equal Remuneration Act, 1976, women must be paid equally to men for the same work.",
    "Can a woman demand workplace creche facilities? -> Under the Maternity Benefit (Amendment) Act, 2017, organizations with more than 50 employees must provide creche facilities."
]


# Step 2: Encode Knowledge Base with SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(legal_texts)

# Step 3: Create FAISS Index for fast retrieval
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Step 4: Load a Lightweight LLM for Answer Generation
generator = pipeline('text-generation', model='tiiuae/falcon-7b-instruct', max_new_tokens=100)

# Step 5: Retrieval + Generation Logic (RAG)
def empowerbot_answer(user_query):
    # Encode user query
    query_embedding = model.encode([user_query])
    
    # Search FAISS index for the most relevant document
    D, I = index.search(np.array(query_embedding), k=1)  # Get top-1 match
    matched_text = legal_texts[I[0][0]]
    
    # Build prompt for LLM
    prompt = f"You are an expert legal assistant for women. Based on the following law: {matched_text} \n\nAnswer the user's question: {user_query}"
    
    # Generate answer using the LLM
    output = generator(prompt)[0]['generated_text']
    
    return output

# Step 6: Feedback Loop (User Ratings)
def feedback_loop(user_query, correct_answer):
    # Store the user's query and the correct answer for manual feedback
    with open('feedback_data.txt', 'a') as f:
        f.write(f"Query: {user_query}\nCorrect Answer: {correct_answer}\n\n")
    
    print("Feedback received, thank you for helping improve EmpowerBot!")

# Step 7: Gradio UI for Chatbot Interface
def chatbot_interface(user_input):
    answer = empowerbot_answer(user_input)
    return answer

# Feedback UI (for rating answers)
def rate_answer(user_input, correct_answer):
    feedback_loop(user_input, correct_answer)
    return "Thank you for your feedback!"

# Gradio Interface (Main Chatbot + Feedback)
iface = gr.Interface(fn=chatbot_interface, 
                     inputs=gr.Textbox(label="Ask your question about women's legal rights"), 
                     outputs=gr.Textbox(label="EmpowerBot's Answer"), 
                     title="EmpowerBot - Women's Legal Assistant", 
                     description="Ask EmpowerBot about women's rights and get reliable legal information.")

feedback_iface = gr.Interface(fn=rate_answer,
                              inputs=[gr.Textbox(label="Your Query"), gr.Textbox(label="Correct Answer")],
                              outputs="text",
                              title="Provide Feedback",
                              description="Help improve EmpowerBot by providing feedback on answers.")

# Launch both interfaces
iface.launch()#main chatbot interface
feedback_iface.launch(share=True)  # Feedback interface for user ratings
