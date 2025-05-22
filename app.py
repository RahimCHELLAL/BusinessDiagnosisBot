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


#Chatbot AI below
os.environ["TRANSFORMERS_NO_TF"] = "1"  # disable tensorflow fallback

model = SentenceTransformer('all-MiniLM-L6-v2')
genai.configure(api_key="AIzaSyDXO1LSxJ9I5281a9E7e6MitdlWFlv_r30")
modelai = genai.GenerativeModel("gemma-3-12b-it")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')



def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

text_path = r'C:\Users\arez3\Desktop\Etudes\Limitless Learning\Gen AI\Mini projet\Books\text\Diagnosisofbusiness.txt'
text = read_text_from_file(text_path)

def chunk_text(text, chunk_size=1000, overlap=200):
    chunk = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk.append(text[start:end])
        start += chunk_size - overlap
    return chunk

def search_similar_chunks(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]



chunks = chunk_text(text)
embeddings = model.encode(chunks)
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))




def trim_text_by_tokens(text, max_tokens):
    # Tokenize text to tokens
    tokens = tokenizer.tokenize(text)
    
    # Trim tokens if longer than max_tokens
    #if len(tokens) > max_tokens:
    #    tokens = tokens[:max_tokens]
    
    # Convert tokens back to string
    trimmed_text = tokenizer.convert_tokens_to_string(tokens)
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
            response = modelai.generate_content(prompt)
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


# Chatbot environment below:
st.set_page_config(page_title="Business Health Chatbot")
st.title("🤖 Business Health Assistant")
user_input = st.text_input("Let me help you, give me the core of yours problems in your buisness, and I will help you get a diagnosis and solutions")


if user_input:
    message = st.chat_message("human")
    message.write(user_input)

    with st.spinner("Thinking..."):
        query = "Give me a diagnose of my buisness problems and possible solutions"
        relevant_chunks = search_similar_chunks(query, model, index, chunks, top_k=1)
        ai_answer = ask_gemini(user_input, relevant_chunks)
        answer = st.chat_message("ai")
        answer.write(ai_answer)
