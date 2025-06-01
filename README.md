# YouTube RAG Assistant

A professional AI-powered chatbot that provides leadership guidance using YouTube video content as a knowledge base. Built with modern RAG (Retrieval-Augmented Generation) architecture using LangChain, Qdrant, and Google Gemini AI.

## Project Overview

This project demonstrates a complete RAG pipeline that:
- Extracts knowledge from YouTube video transcripts
- Creates semantic search capabilities with vector embeddings
- Provides contextual AI responses with source attribution
- Delivers responses through a professional web interface
- Supports multilingual content (Turkish/English)

## Live Demo

**Deployed Application:** [Your Streamlit Cloud URL]

## Architecture

```
YouTube Playlist → Audio Download → Transcription → Vector Store → RAG → Web Interface
     ↓                   ↓             ↓            ↓               ↓         ↓
  pytubefix        OpenAI Whisper    BGE-M3       Qdrant         Gemini AI  Streamlit
```

## Project Structure

```
youtube-rag-assistant/
├── app.py                        # Main Streamlit web application
├── requirements.txt              # Python dependencies
├── config/
│   ├── settings.yaml            # Application configuration
│   └── prompts.yaml            # LLM prompt templates
├── src/
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   └── models.py           # Data models and types
│   └── services/
│       ├── youtube_service.py  # YouTube video downloading
│       ├── transcription_service.py # Audio transcription
│       ├── vector_service.py   # Vector search and embeddings
│       └── rag_service.py     # RAG implementation
├── data/                       # Data storage (gitignored)
└── README.md                  # This file
```

## Implementation Status

### Completed Components
- **YouTube Service**: Download audio from playlists
- **Transcription Service**: Audio-to-text with OpenAI Whisper
- **Vector Service**: Semantic search with LangChain + Qdrant
- **RAG Service**: LLM integration with contextual responses
- **Web Interface**: Professional Streamlit chat application
- **Configuration System**: YAML-based configuration management
- **Data Models**: Type-safe data structures

### Key Features
- **Semantic Search**: Advanced similarity search using BGE-M3 embeddings
- **Source Attribution**: Every response includes video source and confidence score
- **Clickable Links**: Direct access to original YouTube videos
- **Professional UI**: Dark theme with clean, modern design
- **Export Functionality**: Download conversation history
- **Example Questions**: Quick access to common leadership queries

## Technology Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **Streamlit**: Web interface framework
- **LangChain**: RAG framework and document processing
- **Qdrant**: Vector database for semantic search
- **Google Gemini AI**: Large language model for response generation
- **HuggingFace**: Embedding models (BGE-M3)
- **OpenAI Whisper**: Audio transcription

### Supporting Libraries
- **pytubefix**: YouTube video downloading
- **sentence-transformers**: Text embeddings
- **pydantic**: Data validation
- **PyYAML**: Configuration management

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/ezgisubasi/youtube-rag-assistant.git
cd youtube-rag-assistant
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.template .env

# Add your API key
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

### 3. Run the Application
```bash
streamlit run app.py
```

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key
YOUTUBE_PLAYLIST_URL=https://youtube.com/playlist?list=...
```

### Settings (config/settings.yaml)
```yaml
# Model Configuration
model_name: gemini-2.0-flash-exp
embedding_model: altaidevorg/bge-m3-distill-8l

# Vector Database
vector_db_path: data/vector_db
collection_name: youtube_transcripts
retrieval_k: 3

# Processing
whisper_model: medium
language: tr
```

## Data Pipeline (Optional)

If you want to process your own YouTube playlist:

### 1. Download YouTube Audio
```bash
python src/services/youtube_service.py
```

### 2. Transcribe Audio to Text
```bash
python src/services/transcription_service.py
```

### 3. Create Vector Store
```bash
python src/services/vector_service.py
```

### 4. Test Search
```python
from src.services.vector_service import VectorService

service = VectorService()
service.initialize_vector_store()

# Search for content
results = service.search("leadership")
for result in results:
    print(f"{result.video_title}: {result.similarity_score:.3f}")
```

## Usage

### Web Interface
1. Open the Streamlit application
2. Use example questions or type your own
3. View responses with source attribution
4. Click video links to access original content
5. Export conversation history if needed

### Example Queries
- "Nasıl iyi lider olunur?" (How to become a good leader?)
- "Takım çalışması neden önemlidir?" (Why is teamwork important?)
- "Başarılı iş stratejileri nelerdir?" (What are successful business strategies?)

## Key Features Demonstration

### Semantic Search
The system uses advanced embeddings to find relevant content based on meaning, not just keywords.

### Source Attribution
Every response includes:
- Video title and clickable YouTube link
- Confidence score indicating relevance
- Direct access to original source material

### Professional Interface
- Clean, dark theme design
- Real-time response generation
- Conversation history management
- Export functionality

## Deployment

### Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set environment variables in secrets
4. Deploy automatically

### Local Development
```bash
streamlit run app.py
```

## Project Highlights

This project showcases:
- **Modern RAG Architecture**: Complete pipeline from data ingestion to user interface
- **Production-Ready Code**: Clean architecture with proper separation of concerns
- **Professional UI/UX**: Streamlit application with custom styling
- **Multilingual Support**: Handles Turkish and English content
- **Source Transparency**: Full attribution to original video sources
- **Scalable Design**: Modular architecture for easy extension

## Technical Decisions

### Why These Technologies?
- **Qdrant**: High-performance vector database with excellent Python integration
- **BGE-M3**: State-of-the-art multilingual embedding model
- **Gemini AI**: Advanced language model with good Turkish support
- **Streamlit**: Rapid development of professional web interfaces

### Architecture Benefits
- **Modular Design**: Each service can be developed and tested independently
- **Configuration Management**: YAML-based settings for easy deployment
- **Type Safety**: Pydantic models ensure data consistency
- **Error Handling**: Comprehensive error handling throughout the pipeline

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a portfolio project demonstrating modern RAG architecture. Feel free to:
- Fork for your own experiments
- Report issues or suggestions
- Contribute improvements via pull requests

## Contact

**Developer**: Ezgi Subaşı
**Project**: YouTube RAG Assistant Portfolio Project
