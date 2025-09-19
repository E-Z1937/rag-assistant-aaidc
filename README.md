# RAG Assistant - Simple TF-IDF Based Implementation

![RAG System](https://img.shields.io/badge/RAG-Educational-blue?style=for-the-badge&logo=python) ![AAIDC](https://img.shields.io/badge/AAIDC-2025-green?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python) ![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> A **lightweight Retrieval-Augmented Generation (RAG) assistant** built for **AAIDC Project 1**. This implementation demonstrates **core RAG concepts** without heavy dependencies like PyTorch or Hugging Face Transformers.

**✅ No API keys required** • **✅ No GPU / Heavy downloads** • **✅ Works offline** • **✅ Pure Python + scikit-learn + optional FAISS**


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Safety Measures](#safety-measures)
- [Performance](#performance)
- [Evaluation](#evaluation)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

### What This Project Does

This RAG assistant demonstrates retrieval-augmented generation using accessible Python libraries. The system:

1. Loads text documents from a directory
2. Splits them into chunks using sentence boundaries
3. Creates TF-IDF vectors using scikit-learn
4. Performs similarity search with cosine similarity
5. Generates responses using rule-based knowledge + retrieved content

### Educational Purpose

The implementation prioritizes transparency and learning over performance optimization. Every operation includes logging, processing times are tracked, and the code is structured to be easily understood and modified.

### Technical Approach

- **Document Processing**: Sentence-boundary text splitting with configurable chunk size
- **Vectorization**: TF-IDF with 5000 max features, unigrams + bigrams
- **Similarity Search**: Cosine similarity with configurable threshold (0.1 default)
- **Response Generation**: Rule-based responses enhanced with retrieved document content
- **Optional Enhancement**: FAISS integration for improved similarity search

## Features

### Core RAG Pipeline
- 📚 **Document loading and processing** from `.txt` files
- 🔪 **Intelligent text chunking** preserving sentence boundaries  
- 🔢 **TF-IDF vectorization** using scikit-learn optimization
- 🎯 **Similarity search** with cosine similarity
- 📝 **Response generation** combining rules + retrieved content
- 📊 **Source attribution** with similarity scores

### Educational Features
- 🖥️ **Interactive CLI interface** with helpful commands
- ⏱️ **Processing time logging** for performance awareness
- 📜 **Conversation history** tracking within sessions
- 💡 **Example questions** to guide exploration
- 🔍 **Detailed logging** of all operations
- 📂 **Automatic sample documents** for immediate testing

### System Features  
- 🚫 **No external APIs** required - works completely offline
- 💾 **Optional index saving** for faster startup
- 🔧 **Configurable parameters** for chunking and retrieval
- ⚡ **Optional FAISS integration** for performance enhancement
- 🛡️ **Error handling** with educational error messages

## Architecture

### System Pipeline

```
Text Files → Document Loading → Text Chunking → TF-IDF Vectorization → Vector Index
                                                                              ↓
User Query ← Response Generation ← Document Retrieval ← Similarity Search ←──┘
```

### Core Components

#### 1. Document Processing
- **File Loading**: Processes `.txt` files from specified directory
- **Text Splitting**: `SimpleTextSplitter` class chunks text at sentence boundaries
- **Metadata**: Tracks source files, chunk IDs, and processing statistics

#### 2. Vector Processing  
- **TF-IDF Vectorization**: scikit-learn implementation with optimized parameters
- **Sparse Vectors**: Efficient storage and computation for text similarity
- **Vocabulary Management**: Automatic vocabulary building from document collection

#### 3. Retrieval System
- **Similarity Search**: Cosine similarity between query and document vectors
- **Ranking**: Results sorted by similarity score with configurable threshold
- **Optional FAISS**: Enhanced performance for larger document collections

#### 4. Response Generation
- **Rule-Based Knowledge**: Structured information about key topics
- **Context Integration**: Combines rules with retrieved document content
- **Source Attribution**: Tracks which documents contributed to responses

## Installation

### System Requirements

- **Python**: 3.8 or higher (tested on 3.9, 3.10, 3.11)
- **Memory**: 2GB RAM minimum, 4GB recommended  
- **Storage**: 100MB for system files, additional space for documents
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Network**: Not required (works completely offline)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/E-Z1937/rag-assistant-aaidc.git
cd rag-assistant-aaidc

# Install dependencies  
pip install -r requirements.txt

# Run the assistant
python rag_assistant.py
```

### Dependencies

**Required:**
```
numpy>=1.24.0          # Numerical operations
scikit-learn>=1.0.0    # TF-IDF vectorization  
pandas>=2.0.0          # Data handling (minimal usage)
```

**Optional:**
```  
faiss-cpu>=1.7.4       # Enhanced similarity search performance
```

### Virtual Environment Setup

```bash
# Create isolated environment (recommended)
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, sklearn; print('✅ Installation successful!')"
```

## Usage

### Basic Operation

```bash
# Start the system
python rag_assistant.py

# System automatically:
# 1. Creates sample documents (if not exists)
# 2. Loads and processes documents  
# 3. Creates TF-IDF vector index
# 4. Starts interactive prompt
```

### Example Session

```
🤖 Ultra Simple RAG Assistant
===============================================
📁 Creating sample documents...
✅ Created sample documents in 'sample_documents' directory
📚 Loading documents...
INFO: Loaded 3 documents from sample_documents
🔍 Creating vector index...
INFO: Created vector index with 12 chunks

🎯 RAG Assistant is ready!

❓ Ask a question: What is LangChain?

🤖 Answer: LangChain is a framework for developing applications 
powered by language models. Based on the documents: LangChain is a 
powerful framework for developing applications powered by language 
models. It was created to help developers build LLM applications...

⏱️ Processing time: 0.067s

📚 Sources (2 found):
  1. langchain_overview.txt (similarity: 0.734)
     LangChain is a powerful framework for developing applications...
  2. rag_systems_explained.txt (similarity: 0.456)
     RAG systems are widely used for customer support chatbots...
```

### Available Commands

- **Ask questions**: Type any natural language question
- `help` - Show available commands and usage guidance
- `history` - Display previous questions and answers from current session
- `examples` - Show suggested queries for testing the system  
- `quit` - Exit the assistant gracefully

### Sample Questions

The system includes built-in examples that work well with the provided documents:

1. "What is LangChain?"
2. "What are the key components of LangChain?"
3. "Which vector databases are mentioned?"
4. "How does RAG work?"
5. "What are the benefits of RAG systems?"

## Configuration

### Environment Variables

Create a `.env` file for configuration (optional):

```bash
# Document processing
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200

# Retrieval settings  
RAG_SIMILARITY_THRESHOLD=0.1
RAG_MAX_RESULTS=3

# System settings
RAG_LOG_LEVEL=INFO
RAG_SAVE_INDEX=true
```

### Code Configuration

Key parameters can be modified in the code:

#### Text Chunking
```python
text_splitter = SimpleTextSplitter(
    chunk_size=1000,    # Target chunk size in characters
    chunk_overlap=200   # Overlap between consecutive chunks
)
```

#### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    max_features=5000,      # Maximum vocabulary size
    stop_words='english',   # Remove common English words
    ngram_range=(1, 2),     # Include unigrams and bigrams
    min_df=1,              # Minimum document frequency
    max_df=0.95            # Maximum document frequency
)
```

#### Similarity Search
```python
similarity_threshold = 0.1  # Minimum similarity score for results
top_k_documents = 3        # Maximum documents to retrieve per query
```

### Custom Document Collections

To use your own documents:

1. Create a directory with `.txt` files
2. Modify the directory path in the code:
   ```python
   documents = rag.load_documents_from_directory("your_documents_folder")
   ```
3. Run the system - it will automatically process your documents

## Safety Measures

### Input Validation and Sanitization

The system includes several safety features appropriate for educational environments:

**File System Security:**
- Path validation prevents access outside designated directories
- File type restrictions limit processing to `.txt` files
- Error handling for inaccessible or malformed files

**Query Processing Security:**
- Input length validation prevents extremely long queries
- Character encoding handling for consistent text processing
- Error recovery with educational error messages

### Content Filtering

**Document Processing:**
- Validation during document loading to ensure readable content
- Encoding detection and normalization for consistent processing
- Metadata tracking for document provenance and validation

**Response Generation:**
- Rule-based content filtering ensures appropriate educational responses
- Source attribution enables verification of information sources
- Context limits prevent overly long or inappropriate responses

### Privacy Protection

**Data Handling:**
- No permanent storage of user queries or responses
- Session data remains local and is cleared on exit
- Document processing occurs entirely in memory during operation
- Optional index saving stores only processed vectors, not original content

**Logging:**
- System logs focus on performance and operation metrics
- No storage of user-specific information or query content
- Educational logging helps understand system behavior without privacy concerns

### Error Handling and Recovery

**Graceful Degradation:**
- System continues operating even if individual documents fail to load
- Missing dependencies result in feature degradation rather than system failure
- Clear error messages guide users toward resolution

**Educational Error Messages:**
- Error messages explain what went wrong and how to fix it
- System provides guidance for common issues like missing files
- Recovery suggestions help users learn from mistakes

## Performance

### Measured Performance

The system tracks and displays performance metrics for educational purposes:

**Typical Performance (tested on sample documents):**
- Document loading: 0.02-0.05 seconds for 3 sample documents
- Vector index creation: 0.1-0.3 seconds for 8-12 chunks  
- Query processing: 0.05-0.15 seconds per query
- Memory usage: 20-50MB depending on document collection size

**Scalability Characteristics:**
- Document processing scales linearly with collection size
- Memory usage grows with vocabulary size and number of chunks
- Query performance remains consistent regardless of collection size
- TF-IDF approach suitable for collections up to several hundred documents

### Performance Monitoring

The system includes built-in performance tracking:

```python
# Processing time measurement
start_time = datetime.now()
result = rag.ask_question(question)
processing_time_seconds = (datetime.now() - start_time).total_seconds()

# Displayed to user
print(f"⏱️  Processing time: {result['processing_time_seconds']}s")
```

### Optimization Features

**Optional FAISS Integration:**
- Enhanced similarity search performance for larger collections
- Graceful fallback to cosine similarity if FAISS unavailable
- Performance improvement primarily visible with 100+ documents

**Vector Index Persistence:**
- Optional saving of processed vectors to disk
- Faster startup for repeated sessions with same documents  
- JSON format for transparency and debugging

## Evaluation

### Testing Approach

The system includes built-in testing through automatic sample document creation and example queries that demonstrate functionality across different query types.

**Sample Document Collection:**
- 3 comprehensive documents covering LangChain, vector databases, and RAG systems
- Approximately 1,500-2,000 words of technical content
- 8-12 chunks after sentence-boundary processing
- 500-800 unique terms in TF-IDF vocabulary

### Evaluation Metrics

The system tracks several metrics that can be observed during operation:

**Retrieval Metrics:**
- Number of documents retrieved per query
- Similarity scores for retrieved documents  
- Processing time for similarity search
- Vocabulary utilization in queries

**Response Quality Indicators:**
- Source attribution (which documents contributed to response)
- Response length and structure consistency
- Rule-based vs. context-based response portions
- Processing time stability across queries

### Query Testing Examples

**Definition Queries:** Successfully retrieve and combine rule-based definitions with document content
- Example: "What is LangChain?" → rule definition + document details
- Performance: Typically retrieves 2-3 relevant chunks with similarity > 0.4

**Process Queries:** Extract procedural information from documents  
- Example: "How does RAG work?" → step-by-step process from documents
- Performance: Retrieves relevant process descriptions with good context

**Comparison Queries:** Find comparative information in document collection
- Example: "Which vector databases are mentioned?" → list extraction
- Performance: Variable depending on document coverage of comparison topics

### Limitations Assessment

**Current Limitations:**
- Response quality limited by rule-based generation approach
- Vocabulary restricted to terms present in document collection
- No real-time information updates or external knowledge access
- English-only processing without multi-language support

**Performance Boundaries:**
- Optimal performance with document collections under 1MB total size
- TF-IDF approach less effective for very short documents
- Rule-based responses may not cover all possible query types
- Processing time increases linearly with document collection size

## Development

### Code Structure

```
rag-assistant-aaidc/
├── rag_assistant.py      # Main application file (single file implementation)
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
├── LICENSE              # MIT license file
├── .gitignore           # Git ignore rules
└── sample_documents/    # Auto-generated sample documents (created on first run)
    ├── langchain_overview.txt
    ├── vector_databases_guide.txt
    └── rag_systems_explained.txt        
```

### Extension Points

The modular design enables several enhancement opportunities:

**Document Processing Extensions:**
- Add support for PDF, Word, or HTML formats
- Implement different chunking strategies (fixed-size, semantic)
- Add document preprocessing filters or transformations

**Vectorization Enhancements:**
- Integrate dense embeddings (sentence-transformers)
- Implement hybrid sparse/dense retrieval
- Add document classification or clustering

**Response Generation Improvements:**
- Integrate lightweight language models
- Implement template-based responses
- Add response quality assessment

**Interface Enhancements:**  
- Web interface using Gradio or Streamlit
- REST API for programmatic access
- Batch processing capabilities

### Testing and Validation

**Manual Testing Process:**
1. Run system with sample documents
2. Test example queries for expected responses
3. Verify source attribution accuracy
4. Check processing time consistency
5. Test error handling with invalid inputs

**Performance Testing:**
- Document loading with various file sizes
- Vector index creation with different vocabularies
- Query processing under different loads
- Memory usage monitoring during operation

## Contributing

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/E-Z1937/rag-assistant-aaidc.git
cd rag-assistant-aaidc

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Run tests
python rag_assistant.py
# Test with sample queries to verify functionality
```

### Contribution Guidelines

**Code Style:**
- Follow existing code structure and naming conventions
- Include docstrings for new functions and classes
- Add logging for significant operations
- Maintain educational clarity over optimization

**Documentation:**
- Update README.md for new features or changes
- Include example usage for new functionality  
- Document configuration options and parameters
- Maintain accuracy in all claims and descriptions

**Testing:**
- Test new features with sample documents
- Verify backward compatibility with existing functionality
- Include example queries that demonstrate new capabilities
- Document any performance impacts

### Issue Reporting

When reporting issues, please include:
- Python version and operating system
- Complete error messages or logs
- Steps to reproduce the problem
- Expected vs. actual behavior
- Document collection details (if relevant)
