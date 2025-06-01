"""
YouTube RAG Assistant - Streamlit Web Interface
A specialized AI chatbot that provides leadership guidance using YouTube video content.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from services.rag_service import RAGService
    from core.models import RAGResponse
    from core.config import get_config, validate_config
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .assistant-message {
        border-left-color: #28a745;
        background-color: #f8fff9;
    }
    .source-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .confidence-score {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
        color: white;
    }
    .confidence-high { background-color: #28a745; }
    .confidence-medium { background-color: #ffc107; color: #000; }
    .confidence-low { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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
        with st.spinner("🔧 Initializing RAG Service..."):
            service = RAGService()
        return service, None
    except Exception as e:
        return None, str(e)

def get_confidence_class(score: float) -> str:
    """Get CSS class for confidence score."""
    if score >= 0.7:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def format_confidence_score(score: float) -> str:
    """Format confidence score with styling."""
    css_class = get_confidence_class(score)
    return f'<span class="confidence-score {css_class}">{score:.2f}</span>'

def display_message(message: dict):
    """Display a chat message."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    else:
        with st.chat_message("assistant"):
            # Parse response if it contains source info
            if "**Source:**" in content:
                parts = content.split("**Source:**")
                answer = parts[0].strip()
                source_info = parts[1].strip() if len(parts) > 1 else ""

                st.write(answer)

                if source_info:
                    # Extract video title, URL and confidence
                    if "Confidence Score:" in source_info:
                        source_parts = source_info.split("Confidence Score:")
                        video_info = source_parts[0].strip()
                        confidence_str = source_parts[1].strip()

                        try:
                            confidence = float(confidence_str)
                            st.markdown(f"""
                            <div class="source-info">
                                <strong>📹 Source:</strong> {video_info}<br>
                                <strong>🎯 Confidence:</strong> {format_confidence_score(confidence)}
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            st.markdown(f"""
                            <div class="source-info">
                                <strong>📹 Source:</strong> {video_info}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.write(content)

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🎥 YouTube RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered leadership guidance from YouTube video content</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")

        # Configuration validation
        config_valid = validate_config()

        if config_valid:
            st.success("✅ Configuration valid")
        else:
            st.error("❌ Configuration error")
            st.info("Please check your environment variables and settings.")
            return

        # Initialize RAG service
        if not st.session_state.service_initialized:
            service, error = load_rag_service()
            if service:
                st.session_state.rag_service = service
                st.session_state.service_initialized = True
                st.success("✅ RAG Service loaded")
            else:
                st.error(f"❌ Service initialization failed: {error}")
                return
        else:
            st.success("✅ RAG Service ready")

        # System info
        st.subheader("📊 Session Stats")
        st.metric("Conversations", st.session_state.conversation_count)
        st.metric("Messages", len(st.session_state.messages))

        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "Nasıl iyi lider olunur?",
            "Takım çalışması neden önemlidir?",
            "Başarılı iş stratejileri nelerdir?",
            "Liderlik becerileri nasıl geliştirilir?",
            "İnovasyonun önemi nedir?"
        ]

        for question in example_questions:
            if st.button(question, key=f"example_{question}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.rerun()

        # Export conversation
        if st.session_state.messages and st.button("📥 Export Chat", use_container_width=True):
            chat_export = []
            for msg in st.session_state.messages:
                chat_export.append(f"{msg['role'].upper()}: {msg['content']}\\n")

            export_text = "\\n".join(chat_export)
            st.download_button(
                label="Download Chat History",
                data=export_text,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    # Main chat interface
    st.subheader("💬 Chat Interface")

    # Display chat messages
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
        display_message(user_message)

        # Generate response
        if st.session_state.rag_service:
            try:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        start_time = time.time()
                        response = st.session_state.rag_service.generate_response(prompt)
                        end_time = time.time()

                    # Display response
                    if response.answer:
                        st.write(response.answer)

                        # Add response to messages
                        assistant_message = {
                            "role": "assistant",
                            "content": response.answer,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "response_time": f"{end_time - start_time:.2f}s"
                        }
                        st.session_state.messages.append(assistant_message)
                        st.session_state.conversation_count += 1

                        # Show response time
                        st.caption(f"⏱️ Response time: {end_time - start_time:.2f}s")
                    else:
                        st.error("No response generated")

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        else:
            st.error("RAG service not available")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>🎥 YouTube RAG Assistant | Built with Streamlit & LangChain</p>
            <p>Powered by Google Gemini AI & Qdrant Vector Database</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()