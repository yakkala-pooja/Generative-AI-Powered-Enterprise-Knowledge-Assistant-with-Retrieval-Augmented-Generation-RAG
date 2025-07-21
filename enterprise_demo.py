#!/usr/bin/env python3
"""
Enterprise Knowledge Management Demo

This script demonstrates the complete workflow of the enterprise knowledge processor:
1. Document generation and loading
2. Text chunking with RecursiveCharacterTextSplitter
3. Embedding creation (OpenAI Ada or sentence-transformers)
4. FAISS vector store creation and management
5. Semantic search and retrieval
"""

import os
import sys
from typing import Optional
from enterprise_knowledge_processor import EnterpriseKnowledgeProcessor

def interactive_search_demo(processor: EnterpriseKnowledgeProcessor):
    """
    Interactive search demonstration
    """
    print("\nInteractive Search Demo")
    print("=" * 40)
    print("Enter search queries to find relevant enterprise documents.")
    print("Type 'quit' to exit the search demo.\n")
    
    sample_queries = [
        "warehouse inventory management",
        "supplier qualification process", 
        "cold chain temperature monitoring",
        "delivery delay resolution",
        "quality defect investigation",
        "risk assessment framework",
        "customer satisfaction metrics"
    ]
    
    print("Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    print()
    
    while True:
        try:
            query = input("Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                print("Please enter a valid query.\n")
                continue
            
            # Perform search
            results = processor.search_documents(query, k=5)
            
            if not results:
                print("No relevant documents found.\n")
                continue
            
            print(f"\nFound {len(results)} relevant documents:")
            print("-" * 50)
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\nResult {i} (Similarity Score: {score:.3f})")
                print(f"Document: {doc.metadata.get('filename', 'Unknown')}")
                print(f"Type: {doc.metadata.get('document_type', 'Unknown')}")
                print(f"Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
                print(f"Content Preview:")
                print(f"  {doc.page_content[:300]}...")
                print("-" * 50)
            
            print()
            
        except KeyboardInterrupt:
            print("\nSearch demo interrupted.")
            break
        except Exception as e:
            print(f"Error during search: {e}\n")

def analyze_document_corpus(processor: EnterpriseKnowledgeProcessor):
    """
    Analyze the document corpus and show insights
    """
    print("\nDocument Corpus Analysis")
    print("=" * 40)
    
    stats = processor.get_document_statistics()
    
    if "error" in stats:
        print("No documents processed yet.")
        return
    
    # Vector store statistics
    vs_stats = stats['vector_store_stats']
    print(f"Vector Store Statistics:")
    print(f"  Total document chunks: {vs_stats['total_vectors']:,}")
    print(f"  Embedding dimension: {vs_stats['dimension']}")
    print(f"  Embedding model: {vs_stats['embedding_model']}")
    
    # Processing metadata
    if 'processing_metadata' in stats and stats['processing_metadata']:
        metadata = stats['processing_metadata']
        print(f"\nProcessing Information:")
        print(f"  Created: {metadata.get('creation_date', 'Unknown')}")
        print(f"  Chunk size: {metadata.get('chunk_size', 'Unknown')} characters")
        print(f"  Chunk overlap: {metadata.get('chunk_overlap', 'Unknown')} characters")
    
    print(f"\nFile Locations:")
    print(f"  Documents directory: {stats['document_directory']}")
    print(f"  Vector store path: {stats['vector_store_path']}")

def benchmark_search_performance(processor: EnterpriseKnowledgeProcessor):
    """
    Benchmark search performance across different query types
    """
    print("\nSearch Performance Benchmark")
    print("=" * 40)
    
    test_queries = [
        ("Process Query", "warehouse inventory management procedures"),
        ("Quality Query", "supplier qualification and audit process"),
        ("Temperature Query", "cold chain temperature monitoring requirements"), 
        ("Issue Resolution", "delivery delay customer communication"),
        ("Risk Management", "supply chain risk assessment methodology"),
        ("Performance Metrics", "on-time delivery and customer satisfaction"),
        ("Compliance Query", "regulatory requirements and documentation")
    ]
    
    import time
    
    print(f"Testing {len(test_queries)} different query types...\n")
    
    total_time = 0
    all_results = []
    
    for query_type, query in test_queries:
        start_time = time.time()
        results = processor.search_documents(query, k=3)
        search_time = time.time() - start_time
        
        total_time += search_time
        all_results.append((query_type, query, results, search_time))
        
        print(f"{query_type}:")
        print(f"  Query: '{query}'")
        print(f"  Results found: {len(results)}")
        print(f"  Search time: {search_time:.3f} seconds")
        
        if results:
            best_score = min(score for _, score in results)
            print(f"  Best similarity score: {best_score:.3f}")
        
        print()
    
    print(f"Benchmark Summary:")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average time per query: {total_time/len(test_queries):.3f} seconds")
    
    return all_results

def demonstrate_embedding_comparison():
    """
    Compare different embedding models (if available)
    """
    print("\nEmbedding Model Comparison")
    print("=" * 40)
    
    # Test with sentence-transformers
    print("Testing sentence-transformers/all-MiniLM-L6-v2...")
    processor_st = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Check if documents exist, if not generate them
    if not os.path.exists("faiss_index"):
        print("Processing documents with sentence-transformers...")
        stats_st = processor_st.process_all_documents(regenerate=False)
        print(f"Processed {stats_st['total_chunks']} chunks in {stats_st['processing_time_seconds']:.2f} seconds")
    else:
        processor_st.load_vector_store()
        print("Loaded existing sentence-transformers vector store")
    
    # Test search with sentence-transformers
    test_query = "warehouse inventory management procedures"
    results_st = processor_st.search_documents(test_query, k=3)
    
    print(f"\nSentence-Transformers Results for '{test_query}':")
    for i, (doc, score) in enumerate(results_st[:2], 1):
        print(f"  {i}. Score: {score:.3f} - {doc.metadata.get('filename', 'Unknown')}")
    
    # Note about OpenAI comparison
    print(f"\nNote: To compare with OpenAI Ada embeddings:")
    print(f"  1. Set your OpenAI API key as environment variable")
    print(f"  2. Initialize with: EnterpriseKnowledgeProcessor(embedding_model='openai', openai_api_key='your-key')")
    print(f"  3. OpenAI embeddings typically provide higher accuracy but require API calls")

def export_search_results(processor: EnterpriseKnowledgeProcessor, queries: list, output_file: str = "search_results.json"):
    """
    Export search results for analysis
    """
    print(f"\nExporting Search Results to {output_file}")
    print("=" * 40)
    
    import json
    
    export_data = {
        "export_timestamp": processor.get_document_statistics()['processing_metadata'].get('creation_date'),
        "embedding_model": processor.embedding_model_type,
        "queries_and_results": []
    }
    
    for query in queries:
        results = processor.search_documents(query, k=5)
        
        query_data = {
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "filename": doc.metadata.get('filename'),
                    "document_type": doc.metadata.get('document_type'),
                    "chunk_id": doc.metadata.get('chunk_id'),
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:200]
                }
                for doc, score in results
            ]
        }
        export_data["queries_and_results"].append(query_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported search results for {len(queries)} queries to {output_file}")

def main():
    """
    Main demonstration function
    """
    print("Enterprise Knowledge Management System Demo")
    print("=" * 60)
    print("This demo showcases LangChain-based document processing with:")
    print("- RecursiveCharacterTextSplitter for intelligent chunking")
    print("- OpenAI Ada or sentence-transformers embeddings")
    print("- FAISS vector store for efficient similarity search")
    print("=" * 60)
    
    # Initialize processor
    print("\nInitializing Enterprise Knowledge Processor...")
    
    # Check for OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("OpenAI API key found - you can choose between OpenAI or sentence-transformers")
        embedding_choice = input("Use OpenAI embeddings? (y/n, default=n): ").strip().lower()
        
        if embedding_choice == 'y':
            processor = EnterpriseKnowledgeProcessor(
                embedding_model="openai", 
                openai_api_key=openai_key
            )
        else:
            processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    else:
        print("Using sentence-transformers embeddings (no OpenAI API key required)")
        processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Process documents
    print("\nProcessing enterprise documents...")
    stats = processor.process_all_documents(regenerate=False)
    
    print(f"\nProcessing Results:")
    print(f"  Status: {stats['status']}")
    if stats['status'] == 'completed':
        print(f"  Documents processed: {stats['total_documents']}")
        print(f"  Total chunks created: {stats['total_chunks']}")
        print(f"  Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"  Average chunk size: {stats['avg_chunk_size']:.0f} characters")
        
        print(f"\nDocument Types:")
        for doc_type, count in stats['documents_by_type'].items():
            print(f"    {doc_type.replace('_', ' ').title()}: {count}")
    
    # Demo menu
    while True:
        print("\n" + "=" * 60)
        print("Demo Options:")
        print("1. Interactive Search Demo")
        print("2. Document Corpus Analysis")
        print("3. Search Performance Benchmark")
        print("4. Embedding Model Comparison")
        print("5. Export Search Results")
        print("6. View Sample Documents")
        print("7. Exit")
        
        try:
            choice = input("\nSelect an option (1-7): ").strip()
            
            if choice == '1':
                interactive_search_demo(processor)
            elif choice == '2':
                analyze_document_corpus(processor)
            elif choice == '3':
                benchmark_search_performance(processor)
            elif choice == '4':
                demonstrate_embedding_comparison()
            elif choice == '5':
                sample_queries = [
                    "inventory management procedures",
                    "supplier qualification process",
                    "quality control measures",
                    "risk assessment methodology"
                ]
                export_search_results(processor, sample_queries)
            elif choice == '6':
                print("\nSample Documents Generated:")
                if os.path.exists("enterprise_documents"):
                    for filename in os.listdir("enterprise_documents"):
                        print(f"  - {filename}")
                else:
                    print("  No documents directory found")
            elif choice == '7':
                print("Thank you for using the Enterprise Knowledge Management Demo!")
                break
            else:
                print("Invalid option. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main() 