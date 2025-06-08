""" A specialized AI chatbot that provides leadership guidance using YouTube video content."""

# Fix for PyTorch + Streamlit Cloud compatibility
import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import traceback

# Fix PyTorch JIT issues
try:
    import torch
    torch.jit.set_fuser("none")
except:
    pass

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

try:
    from src.services.rag_service import RAGService
    from src.core.models import RAGResponse
    from src.core.config import get_config, validate_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #64b5f6;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #b0b0b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-info {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #4caf50;
        color: #ffffff;
        font-size: 0.9rem;
    }
    .source-info a {
        color: #64b5f6;
        text-decoration: none;
        font-weight: bold;
    }
    .source-info a:hover {
        color: #90caf9;
        text-decoration: underline;
    }
    .stChatMessage {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #ffffff;
    }
    .stChatMessage .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = None
    if "service_initialized" not in st.session_state:
        st.session_state.service_initialized = False
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

@st.cache_resource
def load_rag_service():
    """Load RAG service with caching."""
    try:
        with st.spinner("Initializing RAG Service..."):
            service = RAGService()
        return service, None
    except Exception as e:
        error_msg = f"RAG Service initialization failed: {str(e)}"
        error_trace = traceback.format_exc()
        return None, (error_msg, error_trace)

def display_message(message):
    """Display a chat message."""
    role = message["role"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            content = message["content"]
            sources = message.get("sources", [])
            confidence = message.get("confidence", 0.0)
            
            # Display the main answer
            st.write(content)
            
            # Display source information if available
            if sources and len(sources) > 0:
                source = sources[0]
                video_title = getattr(source, 'video_title', 'Unknown')
                video_url = getattr(source, 'video_url', '#')
                
                st.markdown(f"""
                <div class="source-info">
                    <strong>Source:</strong> {video_title}<br>
                    <strong>Link:</strong> <a href="{video_url}" target="_blank">{video_url}</a><br>
                    <strong>Confidence Score:</strong> {confidence:.2f}
                </div>
                """, unsafe_allow_html=True)

def generate_response(question):
    """Generate response using RAG service and return structured data."""
    if not st.session_state.rag_service:
        return None
    
    try:
        start_time = time.time()
        rag_response = st.session_state.rag_service.generate_response(question)
        end_time = time.time()
        
        # Extract just the answer text (without source formatting)
        answer_text = rag_response.answer
        if "**Source:**" in answer_text:
            answer_text = answer_text.split("**Source:**")[0].strip()
        
        # Create clean message structure
        message = {
            "role": "assistant",
            "content": answer_text,
            "sources": rag_response.sources,
            "confidence": rag_response.confidence_score,
            "response_time": f"{end_time - start_time:.2f}s",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        return message
        
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Error generating response: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">YouTube RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered guidance from YouTube video content!</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("System Status")

        # Configuration validation
        try:
            config_valid = validate_config()
        except Exception as e:
            st.error(f"Configuration validation error: {e}")
            return

        if not config_valid:
            st.error("Configuration error - Please check your API key")
            return

        st.success("Configuration valid")

        # Initialize RAG service
        if not st.session_state.service_initialized:
            with st.spinner("Loading RAG Service..."):
                service, error = load_rag_service()
            
            if service:
                st.session_state.rag_service = service
                st.session_state.service_initialized = True
                st.success("RAG Service loaded")
            else:
                error_msg, error_trace = error if isinstance(error, tuple) else (str(error), "No traceback")
                st.error(f"Service initialization failed: {error_msg}")
                
                # Show detailed error in expander
                with st.expander("Error Details"):
                    st.code(error_trace)
                
                return
        else:
            st.success("RAG Service ready")

        # Session statistics
        st.subheader("Session Stats")
        st.metric("Conversations", st.session_state.conversation_count)
        st.metric("Messages", len(st.session_state.messages))

        # Test RAG System button
        st.subheader("🔧 System Tests")
        if st.button("Test RAG System", use_container_width=True):
            if st.session_state.rag_service:
                try:
                    # Test vector search
                    test_results = st.session_state.rag_service.vector_service.search("test", top_k=1)
                    st.write(f"Vector DB has {len(test_results)} documents")
                    
                    # Test web search
                    web_result = st.session_state.rag_service.web_search_service.search("test")
                    st.write(f"Web search working: {web_result is not None}")
                    
                    # Test simple response
                    response = st.session_state.rag_service.generate_response("Hello")
                    st.write(f"Response: {response.answer[:100]}...")
                    
                except Exception as e:
                    st.error(f"System test error: {e}")
            else:
                st.error("RAG service not initialized")

        # Example questions
        st.subheader("Example Questions")
        example_questions = [
            "Nasıl iyi lider olunur?",
            "Takım çalışması neden önemlidir?",
            "Başarılı iş stratejileri nelerdir?",
            "Liderlik becerileri nasıl geliştirilir?",
            "İnovasyonun önemi nedir?"
        ]

        for question in example_questions:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                # Add user message
                user_message = {
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.messages.append(user_message)
                
                # Generate response
                response_message = generate_response(question)
                if response_message:
                    st.session_state.messages.append(response_message)
                    st.session_state.conversation_count += 1
                
                st.rerun()

        # Control buttons
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.rerun()

        # Export functionality
        if st.session_state.messages:
            if st.button("Export Chat", use_container_width=True):
                chat_export = []
                for msg in st.session_state.messages:
                    chat_export.append(f"{msg['role'].upper()}: {msg['content']}")
                
                export_text = "\n\n".join(chat_export)
                st.download_button(
                    label="Download Chat History",
                    data=export_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    # Main chat interface
    st.subheader("Chat Interface")

    # Display existing messages
    for message in st.session_state.messages:
        display_message(message)

    # Chat input
    if prompt := st.chat_input("Ask a question about leadership or business..."):
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message)

        # Generate response
        with st.spinner("Thinking..."):
            response_message = generate_response(prompt)

        if response_message:
            # Add to message history
            st.session_state.messages.append(response_message)
            st.session_state.conversation_count += 1
            
            # Trigger rerun to display everything properly
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #b0b0b0; padding: 1rem;">
            <p><strong>YouTube RAG Assistant</strong> | Built with Streamlit & LangChain</p>
            <p>Powered by Google Gemini AI & Qdrant Vector Database</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()