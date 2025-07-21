#!/usr/bin/env python3
"""
Quick Start RAG Example

This script demonstrates how to quickly set up and use the Enterprise Knowledge
Management system for Retrieval-Augmented Generation (RAG) applications.
"""

from enterprise_knowledge_processor import EnterpriseKnowledgeProcessor
import os

def basic_rag_example():
    """
    Basic RAG workflow example
    """
    print("Enterprise RAG Quick Start")
    print("=" * 40)
    
    # Step 1: Initialize the processor
    print("1. Initializing processor with sentence-transformers...")
    processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Step 2: Process documents (or load existing)
    print("2. Processing enterprise documents...")
    stats = processor.process_all_documents(regenerate=False)
    
    print(f"   Processed {stats.get('total_chunks', 0)} document chunks")
    print(f"   Using {processor.embedding_model_type} embeddings")
    
    # Step 3: Demonstrate semantic search
    print("\n3. Performing semantic searches...")
    
    # Example queries for enterprise scenarios
    example_queries = [
        "How should we handle warehouse inventory management?",
        "What are the supplier qualification requirements?", 
        "What is the cold chain temperature monitoring process?",
        "How do we resolve delivery delays with customers?",
        "What are the quality control procedures for defects?"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # Retrieve relevant documents
        results = processor.search_documents(query, k=2)
        
        if results:
            print(f"Found {len(results)} relevant documents:")
            
            for j, (doc, score) in enumerate(results, 1):
                print(f"\n  Result {j} (Similarity: {score:.3f}):")
                print(f"    Document: {doc.metadata.get('filename', 'Unknown')}")
                print(f"    Type: {doc.metadata.get('document_type', 'Unknown')}")
                print(f"    Content: {doc.page_content[:150]}...")
        else:
            print("  No relevant documents found")
    
    return processor

def advanced_rag_with_context():
    """
    Advanced RAG with context aggregation
    """
    print("\n\nAdvanced RAG with Context Aggregation")
    print("=" * 50)
    
    processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Load existing vector store if available
    if processor.load_vector_store():
        print("Loaded existing vector store")
    else:
        print("Processing documents...")
        processor.process_all_documents()
    
    # Complex query requiring multiple document types
    complex_query = "What is the complete process for handling a supplier quality issue that affects customer deliveries?"
    
    print(f"\nComplex Query: {complex_query}")
    
    # Retrieve more results for comprehensive context
    results = processor.search_documents(complex_query, k=8)
    
    if results:
        print(f"\nRetrieved {len(results)} relevant documents for context:")
        
        # Group by document type
        context_by_type = {}
        for doc, score in results:
            doc_type = doc.metadata.get('document_type', 'unknown')
            if doc_type not in context_by_type:
                context_by_type[doc_type] = []
            context_by_type[doc_type].append((doc, score))
        
        # Display context by type
        for doc_type, docs in context_by_type.items():
            print(f"\n{doc_type.replace('_', ' ').title()} Documents:")
            for doc, score in docs[:2]:  # Top 2 per type
                print(f"  - {doc.metadata.get('filename')} (Score: {score:.3f})")
                print(f"    {doc.page_content[:100]}...")
        
        # Aggregate context for RAG response
        aggregated_context = "\n\n".join([
            f"From {doc.metadata.get('filename', 'Unknown')}:\n{doc.page_content}"
            for doc, score in results[:4]  # Top 4 most relevant
        ])
        
        print(f"\nAggregated Context Length: {len(aggregated_context)} characters")
        print("This context would be passed to an LLM for generating a comprehensive response.")
        
        # Show sample context structure
        print(f"\nSample Context Structure:")
        print(f"Query: {complex_query}")
        print(f"Context: {aggregated_context[:300]}...")
        print(f"[This would be passed to GPT-4, Claude, or other LLM for final response generation]")

def demonstrate_document_types():
    """
    Show how different document types are processed and retrieved
    """
    print("\n\nDocument Type Analysis")
    print("=" * 40)
    
    processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Load or process documents
    if not processor.load_vector_store():
        processor.process_all_documents()
    
    # Queries specific to each document type
    type_specific_queries = {
        "Standard Operating Procedure": "warehouse receiving and storage procedures",
        "Report": "monthly performance metrics and KPI analysis", 
        "Customer Support Log": "customer complaint resolution and follow-up"
    }
    
    print("Demonstrating retrieval by document type:")
    
    try:
        for doc_type, query in type_specific_queries.items():
            print(f"\n{doc_type} Query: '{query}'")
            
            results = processor.search_documents(query, k=2)  # Reduced from 3 to 2
            
            # Filter and show results by document type
            type_results = [
                (doc, score) for doc, score in results
                if doc_type.lower().replace(' ', '_') in doc.metadata.get('document_type', '')
            ]
            
            if type_results:
                print(f"  Found {len(type_results)} relevant {doc_type.lower()} documents:")
                for doc, score in type_results[:1]:  # Show only top result
                    print(f"    - {doc.metadata.get('filename')} (Score: {score:.3f})")
            else:
                print(f"  No {doc_type.lower()} documents found for this query")
    except Exception as e:
        print(f"Error during document type analysis: {e}")

def export_rag_knowledge_base():
    """
    Export processed knowledge base for external use
    """
    print("\n\nExporting Knowledge Base")
    print("=" * 40)
    
    processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    if not processor.load_vector_store():
        processor.process_all_documents()
    
    # Get statistics
    stats = processor.get_document_statistics()
    
    print("Knowledge Base Statistics:")
    vs_stats = stats['vector_store_stats']
    print(f"  Total vectors: {vs_stats['total_vectors']:,}")
    print(f"  Vector dimension: {vs_stats['dimension']}")
    print(f"  Embedding model: {vs_stats['embedding_model']}")
    
    # Export configuration
    export_config = {
        "knowledge_base_info": {
            "vector_count": vs_stats['total_vectors'],
            "embedding_dimension": vs_stats['dimension'],
            "embedding_model": vs_stats['embedding_model'],
            "vector_store_path": "faiss_index",
            "supported_queries": [
                "Supply chain procedures",
                "Quality management",
                "Risk assessment",
                "Customer support",
                "Performance metrics"
            ]
        },
        "usage_examples": [
            {
                "query": "warehouse inventory procedures",
                "expected_docs": ["sop", "report"],
                "use_case": "Operations training"
            },
            {
                "query": "supplier quality issues",
                "expected_docs": ["sop", "customer_support_log"],
                "use_case": "Quality management"
            }
        ]
    }
    
    import json
    with open("knowledge_base_config.json", "w") as f:
        json.dump(export_config, f, indent=2)
    
    print(f"\nExported knowledge base configuration to knowledge_base_config.json")
    print("This configuration can be used to integrate with external RAG systems")

def main():
    """
    Main function demonstrating all RAG capabilities
    """
    try:
        # Basic RAG example
        processor = basic_rag_example()
        
        # Advanced context aggregation
        advanced_rag_with_context()
        
        # Document type analysis
        demonstrate_document_types()
        
        # Export knowledge base
        export_rag_knowledge_base()
        
        print("\n" + "=" * 60)
        print("RAG Quick Start Complete!")
        print("\nNext Steps:")
        print("1. Run 'python enterprise_demo.py' for interactive features")
        print("2. Integrate with your preferred LLM (GPT-4, Claude, etc.)")
        print("3. Customize document types and processing for your domain")
        print("4. Scale with additional document sources")
        
    except Exception as e:
        print(f"Error during RAG demonstration: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 