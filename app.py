"""
YouTube RAG Assistant - Professional Portfolio Project
A specialized AI chatbot that provides leadership guidance using YouTube video content.
"""

print("🔍 [DEBUG] Starting app.py")

import streamlit as st
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import traceback

print("✅ [DEBUG] Basic imports completed")

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))
print(f"🔍 [DEBUG] Added to sys.path: {src_dir}")

print("🔍 [DEBUG] Starting service imports...")
try:
    from src.services.rag_service import RAGService
    print("✅ [DEBUG] RAGService imported successfully")
except ImportError as e:
    print(f"❌ [DEBUG] RAGService import error: {e}")
    print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
    st.error(f"RAGService import error: {e}")
    st.stop()
except Exception as e:
    print(f"❌ [DEBUG] Unexpected error importing RAGService: {e}")
    print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
    st.error(f"Unexpected error importing RAGService: {e}")
    st.stop()

try:
    from src.core.models import RAGResponse
    print("✅ [DEBUG] RAGResponse imported successfully")
except ImportError as e:
    print(f"❌ [DEBUG] RAGResponse import error: {e}")
    st.error(f"RAGResponse import error: {e}")
    st.stop()

try:
    from src.core.config import get_config, validate_config
    print("✅ [DEBUG] Config functions imported successfully")
except ImportError as e:
    print(f"❌ [DEBUG] Config import error: {e}")
    st.error(f"Config import error: {e}")
    st.stop()

print("✅ [DEBUG] All imports completed successfully")

# Page configuration
print("🔍 [DEBUG] Setting up Streamlit page config...")
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)
print("✅ [DEBUG] Page config set")

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
    .debug-info {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #64b5f6;
        color: #ffffff;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    print("🔍 [DEBUG] Initializing session state...")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = None
    if "service_initialized" not in st.session_state:
        st.session_state.service_initialized = False
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    print("✅ [DEBUG] Session state initialized")

@st.cache_resource
def load_rag_service():
    """Load RAG service with caching and extensive debugging."""
    print("🔍 [DEBUG] load_rag_service called")
    try:
        with st.spinner("Initializing RAG Service..."):
            print("🔍 [DEBUG] About to create RAGService instance...")
            service = RAGService()
            print("✅ [DEBUG] RAGService created successfully")
        return service, None
    except Exception as e:
        error_msg = f"RAG Service initialization failed: {str(e)}"
        error_trace = traceback.format_exc()
        print(f"❌ [DEBUG] {error_msg}")
        print(f"❌ [DEBUG] Full traceback: {error_trace}")
        return None, (error_msg, error_trace)

def display_response_with_source(content, sources):
    """Display response with properly formatted source information."""
    # Display the main answer
    st.write(content)
    
    # Display source information if available
    if sources and len(sources) > 0:
        source = sources[0]  # Get first source
        video_title = getattr(source, 'video_title', 'Unknown')
        video_url = getattr(source, 'video_url', '#')
        confidence = getattr(source, 'similarity_score', 0.0)
        
        st.markdown(f"""
        <div class="source-info">
            <strong>Source:</strong> {video_title}<br>
            <strong>Link:</strong> <a href="{video_url}" target="_blank">{video_url}</a><br>
            <strong>Confidence Score:</strong> {confidence:.2f}
        </div>
        """, unsafe_allow_html=True)

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
            display_response_with_source(content, sources)

def generate_response(question):
    """Generate response using RAG service and return structured data."""
    print(f"🔍 [DEBUG] generate_response called with: '{question}'")
    
    if not st.session_state.rag_service:
        print("❌ [DEBUG] RAG service not available")
        return None
    
    try:
        start_time = time.time()
        print("🔍 [DEBUG] Calling RAG service generate_response...")
        rag_response = st.session_state.rag_service.generate_response(question)
        end_time = time.time()
        print(f"✅ [DEBUG] RAG response received in {end_time - start_time:.2f}s")
        
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
        
        print(f"✅ [DEBUG] Message structure created successfully")
        return message
        
    except Exception as e:
        print(f"❌ [DEBUG] Error in generate_response: {e}")
        print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
        return {
            "role": "assistant",
            "content": f"Error generating response: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

def main():
    """Main application function."""
    print("🔍 [DEBUG] main() function started")
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">YouTube RAG Assistant (Debug Mode)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered guidance from YouTube video content!</p>', unsafe_allow_html=True)

    # Debug information section
    with st.expander("🔧 Debug Information", expanded=False):
        st.markdown(f"""
        <div class="debug-info">
        <strong>System Information:</strong><br>
        - Python version: {sys.version}<br>
        - Current directory: {os.getcwd()}<br>
        - Src directory: {src_dir}<br>
        - Path exists: {src_dir.exists()}<br>
        - Files in current dir: {list(Path('.').iterdir())[:5]}<br>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("System Status")

        # Configuration validation
        print("🔍 [DEBUG] Validating configuration...")
        try:
            config_valid = validate_config()
            print(f"✅ [DEBUG] Configuration validation result: {config_valid}")
        except Exception as e:
            print(f"❌ [DEBUG] Configuration validation failed: {e}")
            st.error(f"Configuration validation error: {e}")
            return

        if not config_valid:
            st.error("Configuration error - Please check your API key")
            return

        st.success("Configuration valid")

        # Initialize RAG service
        if not st.session_state.service_initialized:
            print("🔍 [DEBUG] RAG service not initialized, attempting to load...")
            
            with st.spinner("Loading RAG Service..."):
                service, error = load_rag_service()
            
            if service:
                st.session_state.rag_service = service
                st.session_state.service_initialized = True
                st.success("RAG Service loaded")
                print("✅ [DEBUG] RAG service loaded and cached")
            else:
                error_msg, error_trace = error if isinstance(error, tuple) else (str(error), "No traceback")
                st.error(f"Service initialization failed: {error_msg}")
                
                # Show detailed error in expander
                with st.expander("Error Details"):
                    st.code(error_trace)
                
                print(f"❌ [DEBUG] RAG service initialization failed: {error_msg}")
                return
        else:
            st.success("RAG Service ready")
            print("✅ [DEBUG] RAG service already loaded")

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
                    print("🔍 [DEBUG] Testing vector search...")
                    test_results = st.session_state.rag_service.vector_service.search("test", top_k=1)
                    st.write(f"✅ Vector DB has {len(test_results)} documents")
                    
                    # Test web search
                    print("🔍 [DEBUG] Testing web search...")
                    web_result = st.session_state.rag_service.web_search_service.search("test")
                    st.write(f"✅ Web search working: {web_result is not None}")
                    
                    # Test simple response
                    print("🔍 [DEBUG] Testing simple response...")
                    response = st.session_state.rag_service.generate_response("Hello")
                    st.write(f"✅ Response: {response.answer[:100]}...")
                    
                except Exception as e:
                    print(f"❌ [DEBUG] System test error: {e}")
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
                print(f"🔍 [DEBUG] Example question clicked: {question}")
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
        print(f"🔍 [DEBUG] Chat input received: '{prompt}'")
        
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.messages.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_message = generate_response(prompt)

            if response_message:
                # Display response with source
                display_response_with_source(
                    response_message["content"], 
                    response_message["sources"]
                )
                
                # Add to message history
                st.session_state.messages.append(response_message)
                st.session_state.conversation_count += 1
                
                # Show response time
                st.caption(f"Response time: {response_message['response_time']}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #b0b0b0; padding: 1rem;">
            <p><strong>YouTube RAG Assistant (Debug Mode)</strong> | Built with Streamlit & LangChain</p>
            <p>Powered by Google Gemini AI & Qdrant Vector Database</p>
        </div>
        """,
        unsafe_allow_html=True
    )

print("🔍 [DEBUG] App setup completed")

if __name__ == "__main__":
    print("🔍 [DEBUG] Running main()")
    main()