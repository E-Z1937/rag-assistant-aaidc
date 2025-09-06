#!/usr/bin/env python3
"""
Simple RAG Assistant - AAIDC Project 1
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

# Basic imports - should be available
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import FAISS, fall back to simple similarity if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available, using simple similarity search")

class Document:
    """Simple document class"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleTextSplitter:
    """Simple text splitter - splits on sentences and chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(text_chunks):
                new_metadata = doc.metadata.copy()
                new_metadata['chunk_id'] = i
                chunks.append(Document(page_content=chunk, metadata=new_metadata))
        return chunks

class SimpleRAGSystem:
    """Ultra-simple RAG system using TF-IDF and rule-based responses"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.document_vectors = None
        self.documents = []
        self.text_splitter = SimpleTextSplitter()
        
        # Simple knowledge base for responses
        self.knowledge_base = {
            'langchain': {
                'definition': 'LangChain is a framework for developing applications powered by language models',
                'components': 'Key components include Prompts, Models, Chains, Agents, and Memory',
                'usage': 'Used for building chatbots, QA systems, and LLM applications'
            },
            'vector_database': {
                'definition': 'Vector databases store and query vector embeddings for semantic search',
                'examples': 'Popular ones include FAISS, Chroma, Pinecone, Weaviate, and Qdrant',
                'purpose': 'Enable semantic search and similarity matching for AI applications'
            },
            'rag': {
                'definition': 'Retrieval-Augmented Generation combines LLMs with external knowledge retrieval',
                'process': 'Process: Document ingestion → Embedding → Storage → Retrieval → Generation',
                'benefits': 'Benefits include reduced hallucinations, external knowledge access, and easy updates'
            },
            'faiss': {
                'definition': 'FAISS is Facebook AI Similarity Search library for efficient vector operations',
                'usage': 'Used for similarity search and clustering of dense vectors',
                'features': 'Supports both CPU and GPU operations with various indexing methods'
            }
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all .txt files from directory"""
        documents = []
        
        if not os.path.exists(directory_path):
            self.logger.error(f"Directory not found: {directory_path}")
            return documents
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(
                            page_content=content,
                            metadata={'source': filepath, 'filename': filename}
                        ))
                except Exception as e:
                    self.logger.error(f"Error loading {filepath}: {e}")
        
        self.logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def create_vector_index(self, documents: List[Document]):
        """Create vector index from documents"""
        self.logger.info("Creating vector index...")
        
        # Split documents into chunks
        chunked_docs = self.text_splitter.split_documents(documents)
        self.documents = chunked_docs
        
        # Extract text content
        texts = [doc.page_content for doc in chunked_docs]
        
        # Create TF-IDF vectors
        self.document_vectors = self.vectorizer.fit_transform(texts)
        
        self.logger.info(f"Created vector index with {len(chunked_docs)} chunks")
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Retrieve most relevant documents for query"""
        if self.document_vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def generate_response(self, query: str, context_docs: List[Tuple[Document, float]]) -> str:
        """Generate response based on query and context"""
        query_lower = query.lower()
        
        # Extract context text
        context_texts = [doc.page_content for doc, _ in context_docs]
        context = " ".join(context_texts)
        
        # Rule-based response generation
        for topic, knowledge in self.knowledge_base.items():
            if topic in query_lower:
                # Found relevant topic, combine with context
                if context_docs:
                    return f"{knowledge['definition']}. Based on the documents: {context[:300]}..."
                else:
                    return knowledge['definition']
        
        # If no specific topic matched, use context
        if context_docs:
            # Try to extract key sentences from context
            sentences = context.split('.')[:3]
            return f"Based on the available documents: {'. '.join(sentences).strip()}."
        
        return "I can help answer questions about LangChain, vector databases, RAG systems, and FAISS based on the provided documents."
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Process a question and return response with sources"""
        start_time = datetime.now()
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(question, top_k=4)
        
        # Generate response
        answer = self.generate_response(question, relevant_docs)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': round(score, 3)
                }
                for doc, score in relevant_docs
            ],
            'processing_time_seconds': round(processing_time, 3),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_index(self, filepath: str):
        """Save the vector index to file"""
        try:
            data = {
                'documents': [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in self.documents
                ],
                'vectorizer_vocabulary': dict(self.vectorizer.vocabulary_),
                'created_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Index saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
    
    def load_index(self, filepath: str) -> bool:
        """Load vector index from file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore documents
            self.documents = [
                Document(content=doc['content'], metadata=doc['metadata'])
                for doc in data['documents']
            ]
            
            # Restore vectorizer and vectors
            texts = [doc.page_content for doc in self.documents]
            self.document_vectors = self.vectorizer.fit_transform(texts)
            
            self.logger.info(f"Index loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return False

def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = "sample_documents"
    os.makedirs(sample_dir, exist_ok=True)
    
    documents = {
        "langchain_overview.txt": """
        LangChain is a powerful framework for developing applications powered by language models. 
        It was created to help developers build LLM applications more easily and efficiently.

        Key Components of LangChain:
        1. Prompts - Templates and tools for formatting inputs to language models
        2. Models - Wrappers around different language models from various providers
        3. Chains - Sequences of calls to models and other utilities 
        4. Agents - Systems that use language models to decide which actions to take
        5. Memory - Components for persisting state between chain and agent calls

        LangChain supports integration with many different language models including OpenAI GPT models,
        Hugging Face transformers, and other providers. It's particularly useful for building chatbots,
        question-answering systems, and other applications that combine language models with external data.
        """,
        
        "vector_databases_guide.txt": """
        Vector databases are specialized databases designed to store and query vector embeddings efficiently.
        They have become essential infrastructure for AI applications that require semantic search capabilities.

        Popular Vector Databases:
        - FAISS (Facebook AI Similarity Search): Open-source library for efficient similarity search
        - Pinecone: Managed cloud vector database service
        - Weaviate: Open-source vector search engine with GraphQL interface
        - Chroma: Open-source embedding database focused on simplicity
        - Qdrant: High-performance vector similarity search engine
        - Milvus: Open-source vector database built for scalable similarity search

        Vector databases enable applications like:
        - Semantic search and information retrieval
        - Recommendation systems
        - Retrieval-Augmented Generation (RAG) systems
        - Image and document similarity search
        - Chatbots with knowledge bases
        """,
        
        "rag_systems_explained.txt": """
        Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models 
        with external knowledge retrieval. This approach helps address some key limitations of standalone LLMs.

        The RAG Process:
        1. Document Ingestion: Load and preprocess documents into a knowledge base
        2. Embedding Creation: Convert document chunks into vector representations
        3. Vector Storage: Store embeddings in a vector database for efficient retrieval
        4. Query Processing: When a user asks a question, convert it to a vector
        5. Retrieval: Find the most relevant document chunks using similarity search
        6. Generation: Pass the query and retrieved context to an LLM for response generation

        Benefits of RAG:
        - Reduces hallucinations by grounding responses in real documents
        - Provides access to up-to-date external knowledge
        - More cost-effective than fine-tuning large models
        - Easy to update the knowledge base without retraining
        - Allows for source attribution and fact-checking

        RAG systems are widely used for customer support chatbots, internal knowledge bases,
        research assistance, and any application requiring accurate, source-backed responses.
        """
    }
    
    for filename, content in documents.items():
        filepath = os.path.join(sample_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"✅ Created sample documents in '{sample_dir}' directory")
    return sample_dir

def main():
    """Main CLI interface"""
    print("🤖 Ultra Simple RAG Assistant")
    print("=" * 50)
    print("AAIDC Module 1 Project - Lightweight Implementation")
    print("=" * 50)
    
    # Initialize RAG system
    rag = SimpleRAGSystem()
    
    # Create sample documents if they don't exist
    sample_dir = "sample_documents"
    if not os.path.exists(sample_dir):
        print("📁 Creating sample documents...")
        sample_dir = create_sample_documents()
    
    # Load documents and create index
    print("📚 Loading documents...")
    documents = rag.load_documents_from_directory(sample_dir)
    
    if not documents:
        print("❌ No documents found! Please add .txt files to the sample_documents directory.")
        return 1
    
    print("🔍 Creating vector index...")
    rag.create_vector_index(documents)
    
    # Save index
    rag.save_index("rag_index.json")
    
    print("\n🎯 RAG Assistant is ready!")
    print("You can ask questions about:")
    print("- LangChain framework and components")  
    print("- Vector databases (FAISS, Pinecone, etc.)")
    print("- RAG systems and how they work")
    print("- Or anything in the loaded documents")
    print("\nType 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    # Example questions
    examples = [
        "What is LangChain?",
        "What are the key components of LangChain?",
        "Which vector databases are mentioned?", 
        "How does RAG work?",
        "What are the benefits of RAG systems?"
    ]
    
    print("\n💡 Example questions:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()
    
    # Interactive loop
    conversation_history = []
    
    while True:
        try:
            question = input("❓ Ask a question: ").strip()
            
            if not question:
                continue
            elif question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() in ['help', 'h']:
                print("Commands: 'quit' to exit, 'history' to see past Q&A, 'examples' for sample questions")
                continue
            elif question.lower() == 'history':
                if conversation_history:
                    print("\n📜 Conversation History:")
                    for i, (q, a) in enumerate(conversation_history[-5:], 1):
                        print(f"{i}. Q: {q}")
                        print(f"   A: {a[:100]}...")
                else:
                    print("No conversation history yet.")
                continue
            elif question.lower() == 'examples':
                print("💡 Example questions:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
                continue
            
            # Process question
            print("🔍 Searching...")
            result = rag.ask_question(question)
            
            # Display response
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"⏱️  Processing time: {result['processing_time_seconds']}s")
            
            # Show sources if available
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])} found):")
                for i, source in enumerate(result['sources'][:3], 1):
                    filename = source['metadata'].get('filename', 'Unknown')
                    similarity = source['similarity_score']
                    content_preview = source['content'][:150]
                    print(f"  {i}. {filename} (similarity: {similarity})")
                    print(f"     {content_preview}...")
            
            # Save to history
            conversation_history.append((question, result['answer']))
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
