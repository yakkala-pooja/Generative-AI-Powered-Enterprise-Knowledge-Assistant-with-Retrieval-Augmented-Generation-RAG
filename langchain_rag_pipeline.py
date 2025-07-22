#!/usr/bin/env python3
"""
LangChain RAG Pipeline for Enterprise Knowledge Assistant

This module implements a comprehensive Retrieval-Augmented Generation (RAG) pipeline
using LangChain, OpenAI GPT-4, and FAISS for enterprise knowledge management.
"""

import os
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Local imports
from enterprise_knowledge_processor import EnterpriseKnowledgeProcessor

class LangChainRAGPipeline:
    """
    Comprehensive RAG pipeline using LangChain for enterprise knowledge assistant
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers",
        llm_model: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            openai_api_key: OpenAI API key
            embedding_model: Embedding model type ("openai" or "sentence-transformers")
            llm_model: LLM model name
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens for LLM response
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model_type = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize knowledge processor
        self.knowledge_processor = EnterpriseKnowledgeProcessor(
            embedding_model=embedding_model,
            openai_api_key=self.openai_api_key
        )
        
        # Initialize embeddings
        if embedding_model == "openai" and self.openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize LLM
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name=llm_model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            print("Warning: No OpenAI API key provided. LLM features will be limited.")
            self.llm = None
        
        # Initialize vector store and chains
        self.vector_store = None
        self.qa_chain = None
        self.conversational_chain = None
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Load or create vector store
        self._setup_vector_store()
        self._setup_chains()
    
    def _setup_vector_store(self):
        """Setup or load the FAISS vector store"""
        print("Setting up vector store...")
        
        # Try to load existing vector store
        self.vector_store = self.knowledge_processor.load_vector_store()
        
        if self.vector_store is None:
            print("No existing vector store found. Processing documents...")
            stats = self.knowledge_processor.process_all_documents()
            self.vector_store = self.knowledge_processor.vector_store
            print(f"Created vector store with {stats['total_chunks']} chunks")
        else:
            print("Loaded existing vector store")
    
    def _setup_chains(self):
        """Setup LangChain QA and conversational chains"""
        if not self.llm or not self.vector_store:
            print("Cannot setup chains: Missing LLM or vector store")
            return
        
        print("Setting up LangChain chains...")
        
        # Custom prompt template for enterprise context
        qa_prompt_template = """
You are an expert Enterprise Knowledge Assistant specializing in supply chain management, 
operations, and customer service. Use the following context to provide accurate, detailed, 
and actionable answers.

Context Information:
{context}

Question: {question}

Instructions:
1. Provide a comprehensive answer based on the context
2. If the context contains SOPs, explain the procedures step-by-step
3. If the context contains reports, highlight key metrics and insights
4. If the context contains support logs, explain resolution approaches
5. Include relevant recommendations and best practices
6. If information is incomplete, clearly state what additional information would be helpful
7. Maintain a professional, helpful tone suitable for enterprise users

Answer:"""

        qa_prompt = PromptTemplate(
            template=qa_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Setup QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
        
        # Setup conversational chain for multi-turn conversations
        conversational_prompt_template = """
You are an expert Enterprise Knowledge Assistant. Use the following context and chat history 
to provide informed responses about supply chain operations, quality management, and customer service.

Context: {context}
Chat History: {chat_history}
Human: {question}
Assistant:"""

        conversational_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=conversational_prompt_template
        )
        
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": conversational_prompt}
        )
        
        print("LangChain chains setup complete")
    
    def query(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """
        Query the RAG pipeline for an answer
        
        Args:
            question: User question
            use_conversation: Whether to use conversational chain with memory
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        if not self.llm:
            return self._fallback_retrieval_only(question)
        
        print(f"Processing query: {question}")
        
        try:
            if use_conversation and self.conversational_chain:
                result = self.conversational_chain.invoke({"question": question})
            else:
                result = self.qa_chain.invoke({"query": question})
            
            # Extract and format response
            answer = result.get("answer", result.get("result", ""))
            source_documents = result.get("source_documents", [])
            
            # Process source documents
            sources = []
            seen_files = set()
            for doc in source_documents:
                filename = doc.metadata.get("filename", "Unknown")
                if filename not in seen_files:  # Avoid duplicates
                    seen_files.add(filename)
                    sources.append({
                        "filename": filename,
                        "document_type": doc.metadata.get("document_type", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", -1),
                        "content_preview": doc.page_content[:200] + "...",
                        "relevance": "High"  # Could implement scoring
                    })
            
            return {
                "answer": answer,
                "sources": sources,
                "query": question,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.llm_model,
                "source_count": len(sources)
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return self._fallback_retrieval_only(question)
    
    def _fallback_retrieval_only(self, question: str) -> Dict[str, Any]:
        """
        Fallback method for when LLM is not available - returns retrieved documents only
        """
        print("Using fallback retrieval-only mode")
        
        try:
            results = self.knowledge_processor.search_documents(question, k=5)
            
            sources = []
            context_parts = []
            
            for doc, score in results:
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "document_type": doc.metadata.get("document_type", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", -1),
                    "content_preview": doc.page_content[:200] + "...",
                    "relevance_score": float(score)
                })
                context_parts.append(doc.page_content)
            
            # Create a better formatted answer from retrieved context
            answer = f"Based on the retrieved documents, here are the key insights for your question '{question}':\n\n"
            
            # Group by document type and create structured response
            doc_groups = {}
            seen_files = set()
            unique_sources = []
            
            for source in sources:
                filename = source['filename']
                if filename not in seen_files:  # Avoid duplicates
                    seen_files.add(filename)
                    unique_sources.append(source)
                    doc_type = source['document_type']
                    if doc_type not in doc_groups:
                        doc_groups[doc_type] = []
                    doc_groups[doc_type].append(source)
            
            for doc_type, docs in doc_groups.items():
                answer += f"**{doc_type.replace('_', ' ').title()}:**\n"
                for doc in docs:
                    # Extract key information from the content
                    content = doc['content_preview']
                    # Clean up the content and make it more readable
                    clean_content = content.replace('\n', ' ').strip()
                    if len(clean_content) > 300:
                        clean_content = clean_content[:300] + "..."
                    answer += f"• {clean_content}\n"
                answer += "\n"
            
            # Add a summary note
            answer += f"\n*This response is based on {len(seen_files)} relevant documents. For AI-generated summaries, please provide an OpenAI API key.*"
            
            return {
                "answer": answer,
                "sources": unique_sources,  # Use deduplicated sources
                "query": question,
                "timestamp": datetime.now().isoformat(),
                "model_used": "retrieval_only",
                "source_count": len(unique_sources),
                "note": "This response uses document retrieval only. For AI-generated answers, please provide an OpenAI API key."
            }
            
        except Exception as e:
            return {
                "answer": f"Error retrieving documents: {e}",
                "sources": [],
                "query": question,
                "timestamp": datetime.now().isoformat(),
                "model_used": "error",
                "source_count": 0
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get the conversation history
        
        Returns:
            List of conversation turns
        """
        if not self.memory:
            return []
        
        try:
            history = []
            for message in self.memory.chat_memory.messages:
                history.append({
                    "type": message.type,
                    "content": message.content,
                    "timestamp": datetime.now().isoformat()
                })
            return history
        except:
            return []
    
    def clear_conversation_history(self):
        """Clear the conversation memory"""
        if self.memory:
            self.memory.clear()
            print("Conversation history cleared")
    
    def add_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Add new documents to the knowledge base
        
        Args:
            document_paths: List of paths to new documents
            
        Returns:
            Processing statistics
        """
        print(f"Adding {len(document_paths)} new documents...")
        
        try:
            # Load new documents
            new_documents = self.knowledge_processor.load_documents(document_paths)
            
            # Chunk new documents
            chunked_docs = self.knowledge_processor.chunk_documents(new_documents)
            
            # Add to existing vector store
            if self.vector_store and chunked_docs:
                self.vector_store.add_documents(chunked_docs)
                
                # Save updated vector store
                self.knowledge_processor.save_vector_store(self.vector_store)
                
                return {
                    "status": "success",
                    "documents_added": len(new_documents),
                    "chunks_added": len(chunked_docs),
                    "total_documents": self.vector_store.index.ntotal
                }
            else:
                return {"status": "error", "message": "No vector store or documents to add"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """
        Analyze patterns in user queries for insights
        
        Args:
            queries: List of user queries to analyze
            
        Returns:
            Analysis results
        """
        if not queries:
            return {"error": "No queries provided"}
        
        # Simple pattern analysis
        query_lengths = [len(q.split()) for q in queries]
        
        # Categorize queries by likely intent
        categories = {
            "procedural": ["how", "what", "procedure", "process", "steps"],
            "informational": ["what is", "define", "explain", "describe"],
            "troubleshooting": ["issue", "problem", "error", "fix", "resolve"],
            "metrics": ["performance", "kpi", "metric", "report", "analysis"]
        }
        
        query_categories = {"procedural": 0, "informational": 0, "troubleshooting": 0, "metrics": 0, "other": 0}
        
        for query in queries:
            query_lower = query.lower()
            categorized = False
            
            for category, keywords in categories.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                query_categories["other"] += 1
        
        return {
            "total_queries": len(queries),
            "avg_query_length": sum(query_lengths) / len(query_lengths) if query_lengths else 0,
            "query_categories": query_categories,
            "most_common_category": max(query_categories, key=query_categories.get)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and configuration
        
        Returns:
            System status information
        """
        status = {
            "vector_store_loaded": self.vector_store is not None,
            "llm_available": self.llm is not None,
            "embedding_model": self.embedding_model_type,
            "llm_model": self.llm_model if self.llm else "None",
            "chains_initialized": {
                "qa_chain": self.qa_chain is not None,
                "conversational_chain": self.conversational_chain is not None
            },
            "memory_enabled": self.memory is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.vector_store:
            status["vector_store_stats"] = {
                "total_vectors": self.vector_store.index.ntotal,
                "dimension": self.vector_store.index.d
            }
        
        return status


def main():
    """
    Demonstration of the LangChain RAG pipeline
    """
    print("LangChain RAG Pipeline Demo")
    print("=" * 50)
    
    # Initialize the pipeline
    print("1. Initializing RAG pipeline...")
    rag = LangChainRAGPipeline(
        embedding_model="sentence-transformers",
        llm_model="gpt-4",
        temperature=0.1
    )
    
    # Check system status
    status = rag.get_system_status()
    print("\n2. System Status:")
    print(f"   Vector Store: {'✓' if status['vector_store_loaded'] else '✗'}")
    print(f"   LLM Available: {'✓' if status['llm_available'] else '✗'}")
    print(f"   Embedding Model: {status['embedding_model']}")
    print(f"   LLM Model: {status['llm_model']}")
    
    # Demo queries
    demo_queries = [
        "What are the standard procedures for warehouse inventory management?",
        "How should we handle supplier quality issues?",
        "What are the temperature requirements for cold chain management?",
        "How do we resolve customer delivery delays?",
        "What are the key supply chain performance metrics?"
    ]
    
    print("\n3. Processing Demo Queries:")
    results = []
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        result = rag.query(query)
        results.append(result)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources: {result['source_count']} documents")
        print(f"Model: {result['model_used']}")
    
    # Analyze query patterns
    print("\n4. Query Pattern Analysis:")
    analysis = rag.analyze_query_patterns(demo_queries)
    print(f"   Total Queries: {analysis['total_queries']}")
    print(f"   Average Length: {analysis['avg_query_length']:.1f} words")
    print(f"   Most Common Category: {analysis['most_common_category']}")
    
    print(f"\n   Query Categories:")
    for category, count in analysis['query_categories'].items():
        print(f"     {category}: {count}")
    
    # Demo conversational interaction
    if rag.llm:
        print("\n5. Conversational Demo:")
        conv_queries = [
            "What are warehouse safety procedures?",
            "What about inventory accuracy requirements?",
            "How do these relate to the supplier qualification process?"
        ]
        
        for query in conv_queries:
            print(f"\nUser: {query}")
            result = rag.query(query, use_conversation=True)
            print(f"Assistant: {result['answer'][:150]}...")
    
    print("\n" + "=" * 50)
    print("RAG Pipeline Demo Complete!")
    
    if not rag.llm:
        print("\nNote: Set OPENAI_API_KEY environment variable for full LLM capabilities")
    
    return rag


if __name__ == "__main__":
    pipeline = main() 