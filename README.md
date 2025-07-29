📚 Document Q&A System
A powerful and user-friendly RAG (Retrieval-Augmented Generation) system that allows users to upload any PDF document and ask natural language questions about its contents. Built using LangChain, Gemini, HuggingFace Embeddings, FAISS, and Streamlit.

🔍 Features
✅ Upload any PDF and extract relevant content

💡 Ask questions in natural language

📄 Answers include source references from the document

🧠 Maintains chat history and context

⚡ Fast and lightweight interface built with Streamlit

🔐 Uses environment variables for secure API key management


(You can insert a GIF or screenshot here to show how the app works)

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/document-qa.git
cd document-qa
2. Install Dependencies
Ensure you are using Python 3.8+ and a virtual environment (recommended).

bash
Copy
Edit
pip install -r requirements.txt
3. Set Environment Variables
Create a .env file in the root directory with the following:

env
Copy
Edit
GOOGLE_API_KEY=your_gemini_api_key_here
4. Run the Application
bash
Copy
Edit
streamlit run app.py
📦 File Structure
bash
Copy
Edit
document-qa/
├── main.py                # RAG backend logic
├── app.py                 # Streamlit interface
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
└── README.md              # Project documentation
🧠 How It Works
PDF Upload: User uploads a document via the web interface.

Chunking & Embeddings: The file is split into text chunks, and embeddings are generated using HuggingFace.

Vector Search: FAISS is used to search relevant chunks based on the question.

Answer Generation: Gemini (via LangChain) generates answers using document context.

Source Display: The app shows where each answer was derived from.

📚 Example Questions
Once your PDF is uploaded, try asking:

"What is this document about?"

"Summarize the main findings."

"What are the key recommendations?"

"Who is the intended audience?"

"What topics are covered in Chapter 3?"

🛠️ Built With
Streamlit — UI framework

LangChain — RAG and LLM integration

FAISS — Vector similarity search

Gemini — LLM from Google

HuggingFace Transformers — Embeddings

🔐 Security
Your API keys are managed through a .env file and are never exposed in the UI. Make sure you do not commit your .env file.

