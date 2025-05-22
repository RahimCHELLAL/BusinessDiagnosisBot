#Import libraries below
import streamlit as st
import os


import faiss
import textwrap 
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunk = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk.append(text[start:end])
        start += chunk_size - overlap
    return chunk

# Chatbot environment below:
st.set_page_config(page_title="Business Health Chatbot")
st.title("🤖 Business Health Assistant")


#Chatbot AI below
# Cache all heavy initialization
@st.cache_resource
def load_models():
    # Load all models once
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return {
        'sentence_model': SentenceTransformer('all-MiniLM-L6-v2'),
        'gemini_model': genai.GenerativeModel("gemma-3-12b-it"),
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased')
    }

@st.cache_resource
def process_data():
    # Process text data and create FAISS index once
    text_path = r'Books\text\Diagnosisofbusiness.txt'
    text = read_text_from_file(text_path)
    
    chunks = chunk_text(text)
    embeddings = models['sentence_model'].encode(chunks)
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return chunks, index

# Initialize cached resources
models = load_models()
chunks, index = process_data()


def search_similar_chunks(query, model, index, chunks, top_k=3):
    query_embedding = models['sentence_model'].encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]


def trim_text_by_tokens(text, max_tokens):
    # Tokenize text to tokens
    tokens = models['tokenizer'].tokenize(text)
    
    # Trim tokens if longer than max_tokens
    #if len(tokens) > max_tokens:
    #    tokens = tokens[:max_tokens]
    
    # Convert tokens back to string
    trimmed_text = models['tokenizer'].convert_tokens_to_string(tokens)
    return trimmed_text

def ask_gemini(question, context_chunks, max_context_chars=10000, max_retries=5, initial_wait_time=5):
    context = "\n\n".join(context_chunks)
    context = context[:max_context_chars]  # truncate safely
    context = trim_text_by_tokens(context,1000)
    prompt = f"""Based on the following business knowledge, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"""

    retries = 0
    wait_time = initial_wait_time
    while retries < max_retries:
        try:
            response = models['gemini_model'].generate_content(prompt)
            return response.text
        except ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retries + 1}/{max_retries})")
            print(f"Error details: {e}")
            time.sleep(wait_time)
            retries += 1
            wait_time *= 2  # Exponential backoff
        except Exception as e: # Catch other potential errors
            print(f"An unexpected error occurred: {e}")
            raise # Re-raise other errors
    
    raise ResourceExhausted(f"Failed to get response after {max_retries} retries due to rate limiting.")



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Let me help you, write your business probleme here..."):
    # Add user message to history and display immediately
    st.session_state.messages.append({"role": "human", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

    # Start thinking immediately
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            query = "Give me a diagnose of my business problems and possible solutions"
            relevant_chunks = search_similar_chunks(query, models['sentence_model'], index, chunks, top_k=1)
            ai_answer = ask_gemini(user_input, relevant_chunks)
        
        # Add and display response
        st.session_state.messages.append({"role": "ai", "content": ai_answer})
        st.markdown(ai_answer)

