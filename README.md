Business Health Assistant The Business Health Assistant 
is an intelligent chatbot developed as the final mini project for the Generative AI course by **Limitless Learning (LL)**. 
This assistant provides business diagnostics based on expert-written resources and leverages powerful AI tools for semantic search and natural language understanding. 

## Project Description
This chatbot assists users in diagnosing business issues and exploring management solutions by understanding their questions and retrieving the most relevant information from a curated knowledge base. 
It uses: 

- **Generative AI (Gemini Pro)** to generate insightful responses
- **Semantic search with FAISS and Sentence Transformers** to find relevant context
- **Retrieval-Augmented Generation** from the book _"Diagnosis of Business"_ by **Monica Violeta Achim**
  
##  Technologies Used
- Python 3.10+
- [Streamlit](https://streamlit.io/) for UI framework
- [FAISS](https://github.com/facebookresearch/faiss) for Vector similarity search - [Sentence Transformers](https://www.sbert.net/)
- Text embeddings - [Google Gemini API](https://ai.google.dev) as Language model API

## How to Run the App
1. Clone this repository:
   
       gh repo clone RahimCHELLAL/BusinessDiagnosisBot
       cd business-health-assistant
  
3. Install the required dependencies:
   
       pip install -r requirements.txt
  
5. Add your Gemini API key to a .env file:
   
       GEMINI_API_KEY=your_api_key_here
   
7. Launch the Streamlit app:
   
     streamlit run app.py
   
How It Works The app reads and chunks the business diagnosis text. Each chunk is embedded using Sentence Transformers. 
A FAISS index is built for fast similarity search. When a user enters a query, the system retrieves semantically similar chunks and uses Gemini to generate a response using that context.  


Easter Eggs  Commander Guido might occasionally interrupt with drone-based wisdom, saying things like "Shoganai", "Chi ku", or "Sim extamante", and might joke about attaching flamethrowers to drones.  
Dr. João may offer Stoic insights and strategic reflections inspired by his love for theory, mobile robots, and Europa Universalis.

## Authors
- Eng. Arezki Abderrahim Chellal
- Dr. Fathi Daghrir


Under the supervision of: Pr. Mourad Bouache Mr. Houssam Eddine Boukhalfa 

This project is licensed under the MIT License.
