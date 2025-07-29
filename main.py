"""
RAG System Core Logic
This module handles all the backend processing for document Q&A.
Users don't interact with this directly - it's all behind the scenes.
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini and Chain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Memory imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()


class DocumentQASystem:
    """
    Complete RAG system for document Q&A
    Handles everything behind the scenes - users just ask questions!
    """
    
    def __init__(self):
        """Initialize the Q&A system"""
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = []
        self.document_info = {}
        
        # Check API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize embeddings
        self._setup_embeddings()
        
        # Initialize LLM
        self._setup_llm()
    
    def _setup_embeddings(self):
        """Setup embedding model (internal process)"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def _setup_llm(self):
        """Setup language model (internal process)"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                max_tokens=1000,
                timeout=60,
                max_retries=2,
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini LLM: {str(e)}")
    
    def process_uploaded_document(self, uploaded_file, chunk_size=500, chunk_overlap=200):
        """
        Process user's uploaded document
        
        Args:
            uploaded_file: Streamlit uploaded file object
            chunk_size: Size of text chunks (default: 500)
            chunk_overlap: Overlap between chunks (default: 200)
            
        Returns:
            dict: Processing results for user feedback
        """
        temp_file_path = None
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            # Load PDF document
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("Could not extract content from the PDF file")
            
            # Split into chunks for better processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", "\n"]
            )
            docs = text_splitter.split_documents(documents=documents)
            
            if not docs:
                raise ValueError("Could not create text chunks from the document")
            
            # Create vector database
            self._create_vector_store(docs)
            
            # Setup Q&A chain
            self._setup_qa_chain()
            
            # Store document information
            self.document_info = {
                'filename': uploaded_file.name,
                'pages': len(documents),
                'chunks': len(docs),
                'file_size': len(uploaded_file.getvalue())
            }
            
            # Clear any previous chat history
            self.chat_history = []
            
            return {
                'success': True,
                'message': f"Document processed successfully! Ready to answer questions about {uploaded_file.name}",
                'pages': len(documents),
                'chunks': len(docs)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing document: {str(e)}",
                'pages': 0,
                'chunks': 0
            }
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def _create_vector_store(self, docs):
        """Create vector database from document chunks (internal process)"""
        # Create FAISS index
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        
        # Initialize vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Add documents to vector store
        uuids = [str(uuid4()) for _ in range(len(docs))]
        self.vector_store.add_documents(documents=docs, ids=uuids)
    
    def _setup_qa_chain(self):
        """Setup the question-answering chain (internal process)"""
        if self.vector_store is None:
            raise ValueError("Vector store not created")
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get top 5 relevant chunks
        )
        
        # Create prompt template with memory
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided document. "
            "Use the retrieved context to give accurate and helpful answers. "
            "If you're not sure about something, say so clearly. "
            "Consider the conversation history for context, but always prioritize the document content for facts."
            "\n\n"
            "Document Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        self.qa_chain = create_retrieval_chain(retriever, document_chain)
    
    def ask_question(self, question):
        """
        Process user's question and return answer
        
        Args:
            question (str): User's question
            
        Returns:
            dict: Answer and source information
        """
        if self.qa_chain is None:
            return {
                'success': False,
                'answer': "Please upload a document first before asking questions.",
                'sources': []
            }
        
        if not question or not question.strip():
            return {
                'success': False,
                'answer': "Please ask a valid question.",
                'sources': []
            }
        
        try:
            # Get response from RAG chain
            response = self.qa_chain.invoke({
                "input": question.strip(),
                "chat_history": self.chat_history
            })
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response["answer"]))
            
            # Prepare sources for display
            sources = []
            if "context" in response:
                for doc in response["context"][:3]:  # Show top 3 sources
                    sources.append({
                        "page": doc.metadata.get("page", "Unknown"),
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })
            
            return {
                'success': True,
                'answer': response["answer"],
                'sources': sources
            }
            
        except Exception as e:
            return {
                'success': False,
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.chat_history = []
        return "Conversation history cleared!"
    
    def get_document_info(self):
        """Get information about the processed document"""
        return self.document_info
    
    def get_conversation_length(self):
        """Get number of messages in current conversation"""
        return len(self.chat_history)
    
    def is_ready(self):
        """Check if system is ready to answer questions"""
        return self.qa_chain is not None




