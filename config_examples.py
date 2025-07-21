#!/usr/bin/env python3
"""
Configuration Examples for Enterprise Knowledge Processor

This file demonstrates various configuration options and customizations
for different enterprise use cases.
"""

from enterprise_knowledge_processor import EnterpriseKnowledgeProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Configuration 1: Small Documents (Technical Manuals)
def config_technical_manuals():
    """Configuration optimized for technical manuals and procedures"""
    
    class TechnicalManualProcessor(EnterpriseKnowledgeProcessor):
        def __init__(self, embedding_model="sentence-transformers", openai_api_key=None):
            super().__init__(embedding_model, openai_api_key)
            
            # Smaller chunks for technical content
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,          # Smaller chunks for precise technical info
                chunk_overlap=100,       # Less overlap for technical precision
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Include sentence breaks
            )
        
        def _classify_document_type(self, filename: str) -> str:
            """Custom classification for technical documents"""
            filename_lower = filename.lower()
            if 'manual' in filename_lower or 'guide' in filename_lower:
                return 'technical_manual'
            elif 'spec' in filename_lower or 'specification' in filename_lower:
                return 'specification'
            elif 'procedure' in filename_lower or 'sop' in filename_lower:
                return 'procedure'
            else:
                return super()._classify_document_type(filename)
    
    return TechnicalManualProcessor

# Configuration 2: Large Documents (Reports and Analysis)
def config_large_reports():
    """Configuration optimized for large reports and analytical documents"""
    
    class LargeReportProcessor(EnterpriseKnowledgeProcessor):
        def __init__(self, embedding_model="sentence-transformers", openai_api_key=None):
            super().__init__(embedding_model, openai_api_key)
            
            # Larger chunks for comprehensive reports
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,         # Larger chunks for context
                chunk_overlap=300,       # More overlap for continuity
                length_function=len,
                separators=["\n\n\n", "\n\n", "\n", " ", ""]
            )
        
        def _classify_document_type(self, filename: str) -> str:
            """Custom classification for report documents"""
            filename_lower = filename.lower()
            if 'annual' in filename_lower or 'quarterly' in filename_lower:
                return 'periodic_report'
            elif 'analysis' in filename_lower or 'study' in filename_lower:
                return 'analytical_report'
            elif 'financial' in filename_lower or 'budget' in filename_lower:
                return 'financial_report'
            else:
                return super()._classify_document_type(filename)
    
    return LargeReportProcessor

# Configuration 3: Multilingual Support
def config_multilingual():
    """Configuration with multilingual support"""
    
    class MultilingualProcessor(EnterpriseKnowledgeProcessor):
        def __init__(self, embedding_model="sentence-transformers", openai_api_key=None):
            # Use multilingual embedding model
            if embedding_model == "sentence-transformers":
                from langchain.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                embedding_model = "multilingual-sentence-transformers"
            
            super().__init__(embedding_model, openai_api_key)
            
            # Language-aware text splitting
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", ""]
            )
        
        def _classify_document_type(self, filename: str) -> str:
            """Language-aware document classification"""
            filename_lower = filename.lower()
            
            # Multi-language keywords
            if any(word in filename_lower for word in ['sop', 'procedure', '程序', 'procedimiento']):
                return 'standard_operating_procedure'
            elif any(word in filename_lower for word in ['report', '报告', 'informe']):
                return 'report'
            elif any(word in filename_lower for word in ['support', '支持', 'soporte']):
                return 'customer_support_log'
            else:
                return 'multilingual_document'
    
    return MultilingualProcessor

# Configuration 4: High-Performance Setup
def config_high_performance():
    """Configuration optimized for high-performance scenarios"""
    
    class HighPerformanceProcessor(EnterpriseKnowledgeProcessor):
        def __init__(self, embedding_model="sentence-transformers", openai_api_key=None):
            super().__init__(embedding_model, openai_api_key)
            
            # Optimized chunking for performance
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,          # Balanced size for speed and quality
                chunk_overlap=150,       # Minimal overlap for speed
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
        def create_vector_store(self, documents, batch_size=100):
            """Batch processing for large document sets"""
            print(f"Processing {len(documents)} documents in batches of {batch_size}...")
            
            if not documents:
                raise ValueError("No documents provided")
            
            # Process in batches for memory efficiency
            from langchain.vectorstores import FAISS
            
            vector_store = None
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
            
            self.vector_store = vector_store
            return vector_store
    
    return HighPerformanceProcessor

# Configuration 5: Custom Domain (Legal Documents)
def config_legal_documents():
    """Configuration specialized for legal documents"""
    
    class LegalDocumentProcessor(EnterpriseKnowledgeProcessor):
        def __init__(self, embedding_model="sentence-transformers", openai_api_key=None):
            super().__init__(embedding_model, openai_api_key)
            
            # Legal documents need careful chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,         # Larger chunks to preserve legal context
                chunk_overlap=250,       # More overlap for legal continuity
                length_function=len,
                separators=["\n\n", "\n", ". ", "; ", " ", ""]
            )
        
        def _classify_document_type(self, filename: str) -> str:
            """Legal document classification"""
            filename_lower = filename.lower()
            if 'contract' in filename_lower or 'agreement' in filename_lower:
                return 'contract'
            elif 'policy' in filename_lower or 'regulation' in filename_lower:
                return 'policy_document'
            elif 'compliance' in filename_lower or 'audit' in filename_lower:
                return 'compliance_document'
            elif 'legal' in filename_lower or 'law' in filename_lower:
                return 'legal_reference'
            else:
                return 'legal_document'
        
        def search_documents(self, query: str, k: int = 5, score_threshold: float = 0.8):
            """More conservative similarity threshold for legal precision"""
            return super().search_documents(query, k, score_threshold)
    
    return LegalDocumentProcessor

# Example usage configurations
def demo_configurations():
    """Demonstrate different configurations"""
    
    configs = {
        "Technical Manuals": config_technical_manuals(),
        "Large Reports": config_large_reports(),
        "Multilingual": config_multilingual(),
        "High Performance": config_high_performance(),
        "Legal Documents": config_legal_documents()
    }
    
    print("Available Configurations:")
    print("=" * 40)
    
    for name, processor_class in configs.items():
        print(f"\n{name}:")
        processor = processor_class()
        print(f"  Chunk Size: {processor.text_splitter.chunk_size}")
        print(f"  Chunk Overlap: {processor.text_splitter.chunk_overlap}")
        print(f"  Separators: {processor.text_splitter.separators[:3]}...")
        print(f"  Embedding Model: {processor.embedding_model_type}")

# Environment-specific configurations
class ConfigurationManager:
    """Manage different environment configurations"""
    
    @staticmethod
    def get_development_config():
        """Development environment - fast processing, basic features"""
        return {
            'embedding_model': 'sentence-transformers',
            'chunk_size': 800,
            'chunk_overlap': 150,
            'regenerate_documents': True,
            'enable_caching': False
        }
    
    @staticmethod
    def get_production_config():
        """Production environment - optimized for performance and quality"""
        return {
            'embedding_model': 'openai',  # Higher quality
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'regenerate_documents': False,
            'enable_caching': True,
            'batch_size': 50
        }
    
    @staticmethod
    def get_testing_config():
        """Testing environment - minimal processing for quick tests"""
        return {
            'embedding_model': 'sentence-transformers',
            'chunk_size': 500,
            'chunk_overlap': 100,
            'regenerate_documents': True,
            'document_limit': 5  # Process only 5 documents for testing
        }
    
    @staticmethod
    def create_processor_from_config(config_name: str, openai_api_key=None):
        """Create processor instance from configuration name"""
        
        if config_name == 'development':
            config = ConfigurationManager.get_development_config()
        elif config_name == 'production':
            config = ConfigurationManager.get_production_config()
        elif config_name == 'testing':
            config = ConfigurationManager.get_testing_config()
        else:
            raise ValueError(f"Unknown configuration: {config_name}")
        
        processor = EnterpriseKnowledgeProcessor(
            embedding_model=config['embedding_model'],
            openai_api_key=openai_api_key
        )
        
        # Apply configuration
        processor.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return processor, config

def main():
    """Demonstrate configuration options"""
    print("Enterprise Knowledge Processor - Configuration Examples")
    print("=" * 60)
    
    # Show available configurations
    demo_configurations()
    
    # Show environment configurations
    print("\n\nEnvironment Configurations:")
    print("=" * 40)
    
    for env in ['development', 'production', 'testing']:
        print(f"\n{env.title()} Environment:")
        processor, config = ConfigurationManager.create_processor_from_config(env)
        
        for key, value in config.items():
            if key != 'openai_api_key':  # Don't print API key
                print(f"  {key}: {value}")
    
    print("\n\nUsage Examples:")
    print("=" * 40)
    print("""
# Use technical manual configuration
TechnicalProcessor = config_technical_manuals()
processor = TechnicalProcessor(embedding_model="sentence-transformers")

# Use production environment
processor, config = ConfigurationManager.create_processor_from_config('production')

# Use multilingual configuration
MultilingualProcessor = config_multilingual()
processor = MultilingualProcessor()
""")

if __name__ == "__main__":
    main() 