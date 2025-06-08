""" YouTube RAG Assistant with Simple TTS """

import os
os.environ["TORCH_DISABLE_DYNAMO"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import sys
from pathlib import Path
import time

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

try:
    from src.services.rag_service import RAGService
    from src.services.tts_service import TTSService
    from src.core.config import validate_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .source-info {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #4caf50;
        color: #ffffff;
    }
    .source-info a {
        color: #64b5f6;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "tts_service" not in st.session_state:
    st.session_state.tts_service = None

@st.cache_resource
def load_services():
    """Load RAG and TTS services."""
    try:
        rag_service = RAGService()
        tts_service = TTSService()
        return rag_service, tts_service, None
    except Exception as e:
        return None, None, str(e)

def display_message(message, tts_service, use_tts):
    """Display chat message with optional TTS."""
    role = message["role"]
    
    if role == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            content = message["content"]
            sources = message.get("sources", [])
            confidence = message.get("confidence", 0.0)
            
            # Display answer
            st.write(content)
            
            # TTS button
            if use_tts and tts_service and tts_service.is_available():
                if st.button("🔊 Play", key=f"tts_{hash(content)}"):
                    with st.spinner("Generating speech..."):
                        audio_data = tts_service.generate_speech(content)
                        if audio_data:
                            audio_html = tts_service.create_audio_player(audio_data)
                            st.markdown(audio_html, unsafe_allow_html=True)
                        else:
                            st.error("Failed to generate speech")
            
            # Display source
            if sources:
                source = sources[0]
                video_title = getattr(source, 'video_title', 'Unknown')
                video_url = getattr(source, 'video_url', '#')
                
                st.markdown(f"""
                <div class="source-info">
                    <strong>Source:</strong> {video_title}<br>
                    <strong>Link:</strong> <a href="{video_url}" target="_blank">{video_url}</a><br>
                    <strong>Confidence:</strong> {confidence:.2f}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main app."""
    st.title("🎥 YouTube RAG Assistant")
    st.markdown("AI-powered guidance from YouTube content with voice responses!")

    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Validate config
        if not validate_config():
            st.error("Configuration error - Check your API keys")
            return
        st.success("✅ Configuration valid")
        
        # Load services
        if not st.session_state.rag_service:
            with st.spinner("Loading services..."):
                rag_service, tts_service, error = load_services()
            
            if error:
                st.error(f"Service error: {error}")
                return
            
            st.session_state.rag_service = rag_service
            st.session_state.tts_service = tts_service
            st.success("✅ Services loaded")
        
        # TTS settings
        st.subheader("🔊 Text-to-Speech")
        if st.session_state.tts_service and st.session_state.tts_service.is_available():
            use_tts = st.checkbox("Enable TTS", value=True)
            st.success("✅ TTS Ready")
        else:
            use_tts = False
            st.warning("⚠️ TTS unavailable (API key needed)")
            with st.expander("How to enable TTS"):
                st.markdown("""
                1. Get API key from [ElevenLabs](https://elevenlabs.io/)
                2. Add to Streamlit secrets: `ELEVENLABS_API_KEY = "your_key"`
                3. Restart app
                """)
        
        # Stats
        st.subheader("📊 Stats")
        st.metric("Messages", len(st.session_state.messages))
        
        # Example questions
        st.subheader("💡 Examples")
        examples = [
            "Nasıl iyi lider olunur?",
            "Takım çalışması neden önemlidir?",
            "Başarılı iş stratejileri nelerdir?"
        ]
        
        for i, question in enumerate(examples):
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })
                
                # Generate response
                if st.session_state.rag_service:
                    try:
                        rag_response = st.session_state.rag_service.generate_response(question)
                        answer = rag_response.answer
                        if "**Source:**" in answer:
                            answer = answer.split("**Source:**")[0].strip()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": rag_response.sources,
                            "confidence": rag_response.confidence_score
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}",
                            "sources": [],
                            "confidence": 0.0
                        })
                
                st.rerun()
        
        # Clear button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Chat interface
    st.subheader("💬 Chat")
    
    # Display messages
    for message in st.session_state.messages:
        display_message(message, st.session_state.tts_service, use_tts)
    
    # Chat input
    if prompt := st.chat_input("Ask about leadership or business..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Generate response
        if st.session_state.rag_service:
            with st.spinner("Thinking..."):
                try:
                    rag_response = st.session_state.rag_service.generate_response(prompt)
                    answer = rag_response.answer
                    if "**Source:**" in answer:
                        answer = answer.split("**Source:**")[0].strip()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": rag_response.sources,
                        "confidence": rag_response.confidence_score
                    })
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                        "sources": [],
                        "confidence": 0.0
                    })
        
        st.rerun()

if __name__ == "__main__":
    main()