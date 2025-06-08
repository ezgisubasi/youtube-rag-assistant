""" A specialized AI chatbot that provides leadership guidance using YouTube video content with TTS support."""

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
    from src.services.tts_service import TTSService, StreamlitTTSComponents
    from src.core.models import RAGResponse
    from src.core.config import get_config, validate_config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Assistant with TTS",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling for TTS
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
    .tts-controls {
        margin: 0.5rem 0;
        padding: 0.75rem;
        background-color: #2a2a2a;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b35;
    }
    .tts-button {
        background-color: #ff6b35;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        cursor: pointer;
        margin-right: 0.5rem;
    }
    .tts-button:hover {
        background-color: #e55a2b;
    }
    .audio-player {
        margin: 1rem 0;
        width: 100%;
    }
    .tts-status {
        font-size: 0.9rem;
        color: #b0b0b0;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = None
    if "tts_service" not in st.session_state:
        st.session_state.tts_service = None
    if "service_initialized" not in st.session_state:
        st.session_state.service_initialized = False
    if "tts_initialized" not in st.session_state:
        st.session_state.tts_initialized = False
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    if "tts_settings" not in st.session_state:
        st.session_state.tts_settings = {'enabled': False}

@st.cache_resource(show_spinner=False)
def load_rag_service():
    """Load RAG service with caching."""
    try:
        service = RAGService()
        return service, None
    except Exception as e:
        error_msg = f"RAG Service initialization failed: {str(e)}"
        error_trace = traceback.format_exc()
        return None, (error_msg, error_trace)

@st.cache_resource(show_spinner=False)
def load_tts_service():
    """Load TTS service with caching."""
    try:
        service = TTSService()
        return service, None
    except Exception as e:
        error_msg = f"TTS Service initialization failed: {str(e)}"
        error_trace = traceback.format_exc()
        return None, (error_msg, error_trace)

def display_message_with_tts(message, tts_service, tts_settings):
    """Display a chat message with TTS support."""
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
            
            # TTS controls for assistant messages
            if tts_settings.get('enabled', False) and content and tts_service and tts_service.is_available():
                st.markdown('<div class="tts-controls">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 4])
                
                with col1:
                    if st.button("🔊 Play", key=f"tts_play_{hash(content)}", help="Play this response"):
                        with st.spinner("🎤 Generating speech..."):
                            audio_data = tts_service.generate_speech(
                                text=content,
                                voice_name=tts_settings.get('voice', 'adam')
                            )
                            
                            if audio_data:
                                audio_html = tts_service.create_audio_player(
                                    audio_data, 
                                    autoplay=tts_settings.get('autoplay', False)
                                )
                                with col3:
                                    st.markdown(audio_html, unsafe_allow_html=True)
                                    st.markdown('<div class="tts-status">✅ Audio generated successfully</div>', unsafe_allow_html=True)
                            else:
                                with col3:
                                    st.error("❌ Failed to generate speech")
                
                with col2:
                    # Auto-play for new messages
                    if tts_settings.get('autoplay', False) and message.get('is_new', False):
                        with st.spinner("🎤 Auto-generating speech..."):
                            audio_data = tts_service.generate_speech(
                                text=content,
                                voice_name=tts_settings.get('voice', 'adam')
                            )
                            
                            if audio_data:
                                audio_html = tts_service.create_audio_player(audio_data, autoplay=True)
                                with col3:
                                    st.markdown(audio_html, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display source information if available
            if sources and len(sources) > 0:
                source = sources[0]
                video_title = getattr(source, 'video_title', 'Unknown')
                video_url = getattr(source, 'video_url', '#')
                
                st.markdown(f"""
                <div class="source-info">
                    <strong>📹 Source:</strong> {video_title}<br>
                    <strong>🔗 Link:</strong> <a href="{video_url}" target="_blank">{video_url}</a><br>
                    <strong>📊 Confidence Score:</strong> {confidence:.2f}
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
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_new": True  # Mark as new for auto-play
        }
        
        return message
        
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Error generating response: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "is_new": True
        }

def render_tts_sidebar_controls(tts_service):
    """Render TTS controls in sidebar."""
    st.subheader("🔊 Text-to-Speech")
    
    if not tts_service or not tts_service.is_available():
        st.error("❌ ElevenLabs API key not configured")
        st.info("💡 Add ELEVENLABS_API_KEY to your environment variables")
        
        with st.expander("🔧 How to get ElevenLabs API Key"):
            st.markdown("""
            1. **Sign up at [ElevenLabs](https://elevenlabs.io/)**
            2. **Go to your Profile → API Keys**
            3. **Create a new API key**
            4. **Add it to Streamlit Cloud secrets:**
               ```
               ELEVENLABS_API_KEY = "your_api_key_here"
               ```
            5. **Restart your app**
            """)
        
        return {'enabled': False}
    
    # TTS enabled toggle
    tts_enabled = st.checkbox("🎤 Enable Text-to-Speech", value=False, help="Turn on voice responses")
    
    if not tts_enabled:
        return {'enabled': False}
    
    st.success("✅ TTS Ready")
    
    # Voice selection
    voice_options = {
        '👨 Adam (Male, Deep)': 'adam',
        '👩 Bella (Female, Young)': 'bella',
        '👨 Antoni (Male, Well-rounded)': 'antoni',
        '👩 Elli (Female, Emotional)': 'elli',
        '👨 Josh (Male, Deep)': 'josh',
        '👩 Charlotte (Female, Seductive)': 'charlotte',
        '👩 Matilda (Female, Warm)': 'matilda'
    }
    
    selected_voice_name = st.selectbox(
        "🎭 Voice Selection",
        options=list(voice_options.keys()),
        index=0,
        help="Choose your preferred AI voice"
    )
    selected_voice = voice_options[selected_voice_name]
    
    # Auto-play option
    autoplay = st.checkbox("🔄 Auto-play new responses", value=False, help="Automatically play voice for new responses")
    
    # Test voice button
    if st.button("🎵 Test Voice", help="Test the selected voice"):
        test_text = "Merhaba! Bu ElevenLabs test mesajıdır. Hello! This is an ElevenLabs test message."
        
        with st.spinner("🎤 Generating test audio..."):
            audio_data = tts_service.generate_speech(
                text=test_text,
                voice_name=selected_voice
            )
            
            if audio_data:
                audio_html = tts_service.create_audio_player(audio_data, autoplay=True)
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success("🎉 Voice test successful!")
            else:
                st.error("❌ Voice test failed")
    
    # Usage info
    if st.button("📊 Check API Usage", help="View your ElevenLabs usage"):
        usage_info = tts_service.get_usage_info()
        if usage_info:
            subscription = usage_info.get('subscription', {})
            character_count = subscription.get('character_count', 0)
            character_limit = subscription.get('character_limit', 0)
            
            if character_limit > 0:
                usage_percentage = (character_count / character_limit) * 100
                st.metric(
                    "Characters Used", 
                    f"{character_count:,} / {character_limit:,}",
                    f"{usage_percentage:.1f}%"
                )
                
                # Progress bar
                st.progress(usage_percentage / 100)
                
                if usage_percentage > 80:
                    st.warning("⚠️ Approaching usage limit!")
                elif usage_percentage > 95:
                    st.error("🚨 Usage limit almost reached!")
            else:
                st.info(f"Characters used: {character_count:,}")
        else:
            st.warning("❌ Could not fetch usage info")
    
    return {
        'enabled': True,
        'voice': selected_voice,
        'autoplay': autoplay
    }

def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🎥🔊 YouTube RAG Assistant with TTS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered guidance with voice responses from YouTube content!</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🚀 System Status")

        # Configuration validation
        try:
            config_valid = validate_config()
        except Exception as e:
            st.error(f"Configuration validation error: {e}")
            return

        if not config_valid:
            st.error("❌ Configuration error - Please check your API key")
            return

        st.success("✅ Configuration valid")

        # Initialize RAG service
        if not st.session_state.service_initialized:
            with st.spinner("🔄 Loading RAG Service..."):
                service, error = load_rag_service()
            
            if service:
                st.session_state.rag_service = service
                st.session_state.service_initialized = True
                st.success("✅ RAG Service loaded")
            else:
                error_msg, error_trace = error if isinstance(error, tuple) else (str(error), "No traceback")
                st.error(f"❌ Service initialization failed: {error_msg}")
                
                # Show detailed error in expander
                with st.expander("🔍 Error Details"):
                    st.code(error_trace)
                
                return
        else:
            st.success("✅ RAG Service ready")

        # Initialize TTS service
        if not st.session_state.tts_initialized:
            with st.spinner("🔄 Loading TTS Service..."):
                tts_service, tts_error = load_tts_service()
            
            if tts_service:
                st.session_state.tts_service = tts_service
                st.session_state.tts_initialized = True
                
                if tts_service.is_available():
                    st.success("✅ TTS Service ready")
                else:
                    st.warning("⚠️ TTS Service loaded (API key needed)")
            else:
                st.warning("⚠️ TTS Service failed to load")

        # TTS Controls Section
        st.markdown("---")
        if st.session_state.tts_service:
            st.session_state.tts_settings = render_tts_sidebar_controls(st.session_state.tts_service)
        else:
            st.session_state.tts_settings = {'enabled': False}

        # Session statistics
        st.markdown("---")
        st.subheader("📊 Session Stats")
        st.metric("💬 Conversations", st.session_state.conversation_count)
        st.metric("📝 Messages", len(st.session_state.messages))

        # System tests
        st.subheader("🔧 System Tests")
        if st.button("🧪 Test RAG System", use_container_width=True):
            if st.session_state.rag_service:
                try:
                    # Test vector search
                    test_results = st.session_state.rag_service.vector_service.search("test", top_k=1)
                    st.write(f"✅ Vector DB: {len(test_results)} documents")
                    
                    # Test web search
                    web_result = st.session_state.rag_service.web_search_service.search("test")
                    st.write(f"✅ Web search: {'Working' if web_result else 'Failed'}")
                    
                    # Test simple response
                    response = st.session_state.rag_service.generate_response("Hello")
                    st.write(f"✅ Response: {response.answer[:50]}...")
                    
                except Exception as e:
                    st.error(f"❌ System test error: {e}")
            else:
                st.error("❌ RAG service not initialized")

        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "Nasıl iyi lider olunur?",
            "Takım çalışması neden önemlidir?",
            "Başarılı iş stratejileri nelerdir?",
            "Liderlik becerileri nasıl geliştirilir?",
            "İnovasyonun önemi nedir?"
        ]

        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
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
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            st.rerun()

        # Export functionality
        if st.session_state.messages:
            if st.button("💾 Export Chat", use_container_width=True):
                chat_export = []
                for msg in st.session_state.messages:
                    timestamp = msg.get('timestamp', 'Unknown')
                    chat_export.append(f"[{timestamp}] {msg['role'].upper()}: {msg['content']}")
                
                export_text = "\n\n".join(chat_export)
                st.download_button(
                    label="📄 Download Chat History",
                    data=export_text,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    # Main chat interface
    st.subheader("💬 Chat Interface")

    # Display existing messages with TTS support
    for message in st.session_state.messages:
        # Remove 'is_new' flag for existing messages
        if 'is_new' in message:
            message['is_new'] = False
        
        display_message_with_tts(message, st.session_state.tts_service, st.session_state.tts_settings)

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
        with st.spinner("🤔 Thinking..."):
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
            <p><strong>🎥 YouTube RAG Assistant with TTS</strong> | Built with Streamlit & LangChain</p>
            <p>Powered by Google Gemini AI, Qdrant Vector Database & ElevenLabs TTS</p>
            <p style="font-size: 0.8rem;">🔊 Voice responses powered by ElevenLabs AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()