"""
Simple Document Q&A Interface
Users just upload documents and ask questions - everything else is automatic!
"""

import streamlit as st
import os
from main import DocumentQASystem

# Page setup
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simple, clean styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E7D32;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .upload-section {
        #    set a pleasing background color
        background-color: #f0f4f8;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        text-align: center;
        margin: 2rem 0;
    }
    .chat-container {
        background-color: black;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #eee;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        # set a light background for info boxes 
        background-color: black;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .source-box {
        background-color: black;
        border-left: 3px solid #4caf50;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    """Initialize the Q&A system"""
    if 'qa_system' not in st.session_state:
        try:
            with st.spinner("Setting up the system..."):
                st.session_state.qa_system = DocumentQASystem()
            return True
        except Exception as e:
            st.error(f"❌ Setup failed: {str(e)}")
            if "GOOGLE_API_KEY" in str(e):
                st.info("💡 Please make sure you have set your GOOGLE_API_KEY in a .env file")
            return False
    return True


def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-title">📚 Ask Questions About Your Documents</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload any PDF and start asking questions - it\'s that simple!</p>', unsafe_allow_html=True)


def display_upload_section():
    """Display document upload section"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    if 'document_processed' not in st.session_state:
        st.markdown("### 📄 Upload Your Document")
        st.markdown("Choose a PDF file to get started")
        
        uploaded_file = st.file_uploader(
            "Select PDF file",
            type=['pdf'],
            help="Upload a PDF document to ask questions about its content",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            process_document(uploaded_file)
    
    else:
        doc_info = st.session_state.qa_system.get_document_info()
        st.markdown(f"### ✅ Document Ready: {doc_info['filename']}")
        st.markdown(f"📊 **{doc_info['pages']} pages** processed into **{doc_info['chunks']} sections**")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📄 Upload Different Document", use_container_width=True):
                reset_session()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.qa_system.clear_conversation()
                st.session_state.messages = []
                st.success("Conversation cleared!")
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def process_document(uploaded_file):
    """Process the uploaded document"""
    with st.spinner(f"📖 Reading and processing {uploaded_file.name}..."):
        result = st.session_state.qa_system.process_uploaded_document(uploaded_file)
    
    if result['success']:
        st.markdown(f"""
        <div class="success-message">
            <strong>🎉 Success!</strong><br>
            {result['message']}<br>
            <small>📄 {result['pages']} pages • 📝 {result['chunks']} sections • Ready for questions!</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.document_processed = True
        st.session_state.messages = []
        st.rerun()
    else:
        st.error(f"❌ {result['message']}")


def display_chat_interface():
    """Display the main chat interface"""
    if 'document_processed' not in st.session_state:
        return
    
    st.markdown("### 💬 Ask Your Questions")
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi! I've processed your document and I'm ready to answer questions about it. What would you like to know?",
            "sources": []
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 See where this information came from", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>📖 Source {i} (Page {source['page']}):</strong><br>
                            <em>"{source['content']}"</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if question := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(question)
        
        # Get and display response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Looking for the answer..."):
                response = st.session_state.qa_system.ask_question(question)
            
            if response['success']:
                st.write(response['answer'])
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources']
                })
                
                # Show sources if available
                if response['sources']:
                    with st.expander("📚 See where this information came from", expanded=False):
                        for i, source in enumerate(response['sources'], 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>📖 Source {i} (Page {source['page']}):</strong><br>
                                <em>"{source['content']}"</em>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error(response['answer'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": []
                })


def display_help_section():
    """Display help and example questions"""
    if 'document_processed' not in st.session_state:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 How it works:</h3>
            <ol>
                <li><strong>Upload</strong> your PDF document above</li>
                <li><strong>Wait</strong> a moment while it's processed</li>
                <li><strong>Ask</strong> any questions about the content</li>
                <li><strong>Get</strong> answers with source references</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Example questions you can ask:")
        examples = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "Who are the main people mentioned?",
            "What recommendations are made?"
        ]
        
        for example in examples:
            st.markdown(f"• {example}")
    
    else:
        # Show conversation stats
        conv_length = st.session_state.qa_system.get_conversation_length()
        if conv_length > 0:
            st.markdown(f"""
            <div class="info-box">
                💭 <strong>Conversation:</strong> {conv_length} messages<br>
                🧠 <strong>Memory:</strong> I remember our conversation context!
            </div>
            """, unsafe_allow_html=True)


def reset_session():
    """Reset the entire session"""
    keys_to_keep = []  # Keep nothing, reset everything
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]


def main():
    """Main application"""
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Display header
    display_header()
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Upload section
        display_upload_section()
        
        # Chat interface
        display_chat_interface()
    
    with col2:
        # Help section
        display_help_section()
        
        # System status (minimal)
        with st.expander("ℹ️ System Info", expanded=False):
            if st.session_state.qa_system.is_ready():
                st.success("✅ Ready to answer questions")
            else:
                st.info("⏳ Upload a document to get started")
            
            api_status = "✅ Connected" if os.getenv("GOOGLE_API_KEY") else "❌ Not configured"
            st.write(f"**API Status:** {api_status}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #999; font-size: 0.9rem;'>"
        "Powered by AI • Built with Streamlit • Simple document Q&A"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()