#Import libraries below
import streamlit as st
import os

import faiss
import textwrap 
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

import time
import random
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted

#Hello !
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=1000, overlap=200): #TODO: better chunking, by extending the limit untill the phrase
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
    
    return {
        'sentence_model': SentenceTransformer('all-MiniLM-L6-v2'),
        #'gemini_model': genai.GenerativeModel("gemma-3-12b-it"),
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'client' : genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    }

@st.cache_resource
def process_data():
    # Process text data and create FAISS index once
    text_path = r'Books/text/Diagnosisofbusiness.txt' # For Linux based systems
    #text_path = r'Books\text\Diagnosisofbusiness.txt'# For Windows based systems
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

def ask_gemini(question, context_chunks, instruction, max_context_chars=10000, max_retries=5, initial_wait_time=5):
    context = "\n\n".join(context_chunks)
    context = context[:max_context_chars]  # truncate safely
    context = trim_text_by_tokens(context,1000)
    prompt = f"""Based on the following business knowledge, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"""

    retries = 0
    wait_time = initial_wait_time
    while retries < max_retries:
        try:
            #config = genai.GenerationConfig(system_instruction=instruction)

            response = models['client'].models.generate_content(
                model = "gemini-2.0-flash",
                config = types.GenerateContentConfig(
                system_instruction=instruction),
                contents=prompt
                
            )
            
            #models['gemini_model'].generate_content(prompt)
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


SAMPLE_QUESTIONS = [
    "How can I improve my customer retention?",
    "What are signs of financial distress in a business?",
    "How to evaluate my marketing strategy effectiveness?",
    "What operational inefficiencies should I look for?",
    "How can I better manage my cash flow?",
    "What does Miles (2000) say about the purpose of diagnosis? ",
    "According to Niculescu (1998), what does business analysis involve?",
    "What is the usefulness of a business diagnosis for a manager?",
    "What is \"diagnosis\" in the context of a business company?",
    "How to asses Product responsibility ?"
]





if "messages" not in st.session_state:
    st.session_state.messages = []

# Desplay a random suggestion
if len(st.session_state.messages) == 0:
    
    suggestion = SAMPLE_QUESTIONS[random.randint(0, len(SAMPLE_QUESTIONS)-1)]
    with st.chat_message("ai"):
        st.markdown(f"💡 **Try asking:** *{suggestion}*")



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# About section
with st.expander("🤖 About the Business Health Assistant", expanded=False):
    st.markdown("""
    This chatbot was created as part of the Final Mini Project of the Generative AI course offered by Limitless Learning (LL).

    It uses:
    - Generative AI (Gemini Pro).
    - Based on Retrived Augmented Generation (RAG) pipeline with chunking, Semantic search with FAISS, and Sentence Transformers,
    - Based on books:
                
                Diagnosis of business, Monica Violeta Achim 

    Created by **Eng. Arezki Abderrahim Chellal** and **Dr. Fathi Daghrir**. 
                
    Under the course supervision of **Pr. Mourad Bouache** and **Houssam Eddine Boukhalfa**.
                
    ---  
    *P.S. Commander Guido might occasionally interrupt with drone-based wisdom, or Dr.João with his stoicism phylosophy. (Easter egg)*
    """)

if user_input := st.chat_input("Let me help you, describe your business challenges..."):

    # Add user message to history and display immediately
    st.session_state.messages.append({"role": "human", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)

        
    # Start thinking immediately
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # Easter egg: Guido
            if "guido" in user_input.lower():
                query = user_input
                instruction = """Your name is Guido Berger, also known as Commander Guido,  
                - you are Brazilian, from Rio Grande de Sul, and speak english and portuguese and understand all languages,
                - you are a random person and you like to finish the phrases randomly using words such as shonganai, Sim or Exatamente,
                - you love drones and works actively with them and you generally joke about putting flamethrower on them."""

                
            # Easter egg: João
            elif  "joao" in user_input.lower() or "joão" in user_input.lower():
                query = user_input
                instruction = """Your name is João Braun,
                 - you are a person that love stoicism phylosophy,
                 - you work with mobile robots and love playing strategic games such as Europa Universalis 4,
                 - you love understanding and explaining the theory behind every aspect in your research,
                 - you love crossfit and love to talk bout crossfit."""
                
            # Normal response
            else:
                query = user_input #"Give me a diagnose of my business problems and possible solutions"
                instruction = """You are a helpful business assistant AI. Your job is to help users by only using the provided context.,
                - You are Expert in Business Evaluation and have a Ph.D. in Business Diagnosis,
                - You understand all languages.
                """
            
            relevant_chunks = search_similar_chunks(query, models, index, chunks, top_k=1)
            ai_answer = ask_gemini(user_input, relevant_chunks, instruction)
        
        # Add and display response
        st.session_state.messages.append({"role": "ai", "content": ai_answer})
        st.markdown(ai_answer)

