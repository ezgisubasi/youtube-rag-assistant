# youtube-rag-assistant
# YouTube RAG Assistant

A specialized AI chatbot that provides leadership guidance using YouTube video content as a knowledge base. Built with LangChain, Qdrant, and Gemini AI.

## 🎯 Project Overview

This project demonstrates modern RAG (Retrieval-Augmented Generation) architecture by:
- Extracting knowledge from YouTube video transcripts
- Creating semantic search capabilities with vector embeddings  
- Providing contextual AI responses with source attribution
- Supporting multilingual content (Turkish/English)

## 🏗️ Architecture

```
YouTube Playlist → Audio Download → Transcription → Vector Store → RAG → Chat Interface
     ↓                   ↓             ↓            ↓               ↓         ↓
  pytubefix        OpenAI Whisper    BGE-M3       Qdrant         Gemini AI  Streamlit
```

## 📁 Project Structure

```
youtube-rag-assistant/
├── config/
│   ├── settings.yaml          # Application configuration
│   └── prompts.yaml          # LLM prompt templates
├── src/
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   ├── models.py         # Data models and types
│   │   └── utils.py          # Utility functions
│   └── services/
│       ├── youtube_service.py    # YouTube video downloading
│       ├── transcription_service.py  # Audio transcription
│       ├── vector_service.py     # Vector search and embeddings
│       └── rag_service.py       # RAG implementation (TBD)
├── data/                     # Data storage (gitignored)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```


## ✅ Current Implementation Status

### Completed Components:
- ✅ **YouTube Service**: Download audio from playlists
- ✅ **Transcription Service**: Audio-to-text with Whisper
- ✅ **Vector Service**: Semantic search with LangChain + Qdrant
- ✅ **Configuration System**: YAML-based configuration
- ✅ **Data Models**: Type-safe data structures

### In Progress:
- 🔄 **RAG Service**: LLM integration with context
- 🔄 **Streamlit Interface**: Web chat application
- 🔄 **Voice Output**: Text-to-speech integration

## 🚀 Quick Start

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

### 3. Run Pipeline
```bash
# Download YouTube audio
python src/services/youtube_service.py

# Transcribe audio to text
python src/services/transcription_service.py

# Create vector store
python src/services/vector_service.py
```

### 4. Test Search
```python
from src.services.vector_service import VectorService

service = VectorService()
service.create_vector_store()

# Search for content
results = service.search("sürdürülebilirlik")
for result in results:
    print(f"{result.video_title}: {result.similarity_score:.3f}")
```

## 🛠️ Tech Stack

### Core Technologies:
- **Python 3.10+**: Main programming language
- **LangChain**: RAG framework and document processing
- **Qdrant**: Vector database for semantic search
- **HuggingFace**: Embedding models (BGE-M3)
- **OpenAI Whisper**: Audio transcription
- **Google Gemini**: Large language model

### Supporting Libraries:
- **pytubefix**: YouTube video downloading
- **sentence-transformers**: Text embeddings
- **streamlit**: Web interface (planned)
- **pydantic**: Data validation
- **PyYAML**: Configuration management

## 🔧 Configuration

### Environment Variables:
```bash
GEMINI_API_KEY=your_gemini_api_key
YOUTUBE_PLAYLIST_URL=https://youtube.com/playlist?list=...
```

### Settings (config/settings.yaml):
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

## 📊 Key Features

### 🎯 **Specialized Knowledge Base**
- Extracts insights from leadership/business YouTube content
- Maintains source attribution for transparency
- Supports multilingual content processing

### 🔍 **Advanced Search**
- Semantic similarity search (not just keywords)
- Configurable relevance thresholds
- Chunk-based retrieval for precise context

### 🤖 **Modern AI Stack**
- State-of-the-art embedding models
- Efficient vector storage and retrieval
- Ready for LLM integration

## 🧪 Testing

### Test Vector Service:
```python
# Run comprehensive tests
python src/services/vector_service.py

# Test specific queries
from src.services.vector_service import VectorService
service = VectorService()
service.create_vector_store()

# Results 
results = service.search("sürdürülebilirlik")  # Turkish
```

## 🎯 Next Steps

1. **RAG Service Integration**: Connect vector search with Gemini AI
2. **Streamlit Interface**: Build interactive chat interface
3. **Voice Output**: Add text-to-speech capabilities
4. **Web Deployment**: Deploy to Streamlit Cloud or Heroku
5. **Evaluation Metrics**: Add retrieval and generation quality metrics

## 🤝 Contributing

This is a portfolio/research project demonstrating modern RAG architecture. Feel free to:
- Report issues or suggestions
- Fork for your own experiments
- Contribute improvements via pull requests

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎥 Demo

*Demo video and live deployment links coming soon...*
