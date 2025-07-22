# Enterprise Knowledge Assistant with Retrieval-Augmented Generation (RAG)

A comprehensive enterprise knowledge management system that combines document search, business intelligence, and natural language processing to provide intelligent answers from both unstructured documents and structured data.

## Features

### Document Q&A (RAG Pipeline)
- **Semantic Search**: Find relevant documents using advanced embeddings
- **Context-Aware Answers**: Generate intelligent responses with source citations
- **Conversational Memory**: Maintain context across multiple questions
- **Document Processing**: Support for various document formats (TXT, PDF, DOCX)

### Business Intelligence (SQL Agent)
- **KPI Dashboard**: Pre-built queries for common business metrics
- **Natural Language Queries**: Ask questions in plain English
- **Custom SQL Execution**: Run custom database queries
- **Data Visualization**: Interactive charts and graphs

### Multi-Interface Access
- **Streamlit Dashboard**: User-friendly web interface
- **FastAPI Backend**: RESTful API for integration
- **Command Line**: Interactive terminal interface
- **Programmatic Access**: Python SDK for custom applications

### Enterprise-Ready Features
- **Scalable Architecture**: Modular design for easy extension
- **Security**: API key authentication and input validation
- **Monitoring**: Usage analytics and performance metrics
- **Documentation**: Comprehensive API docs and examples

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for full functionality)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd enterprise-knowledge-assistant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Run the demo**
```bash
python run_enterprise_assistant.py demo
```

### Usage Options

#### 1. **Comprehensive Demo**
```bash
python run_enterprise_assistant.py demo
```
- Tests all features
- Generates performance report
- Shows system capabilities

#### 2. **Web Dashboard**
```bash
python run_enterprise_assistant.py streamlit
```
- Access at: http://localhost:8501
- Interactive document Q&A
- KPI dashboard with charts
- Real-time data visualization

#### 3. **API Server**
```bash
python run_enterprise_assistant.py api
```
- RESTful API endpoints
- Documentation: http://localhost:8000/docs
- Integration-ready for applications

#### 4. **Interactive Mode**
```bash
python run_enterprise_assistant.py interactive
```
- Command-line interface
- Step-by-step guidance
- Testing individual components

#### 5. **All Services**
```bash
python run_enterprise_assistant.py all
```
- Starts both Streamlit and FastAPI
- Full system access
- Development environment

## Project Structure

```
enterprise-knowledge-assistant/
├── Core Components
│   ├── langchain_rag_pipeline.py    # RAG pipeline implementation
│   ├── sql_agent.py                 # SQL agent for database queries
│   ├── enterprise_knowledge_processor.py  # Document processing
│   └── contoso_abf_loader.py        # Data loading utilities
├── Interfaces
│   ├── streamlit_dashboard.py       # Web dashboard
│   ├── fastapi_backend.py           # REST API server
│   └── enterprise_assistant_demo.py # Demo and CLI interface
├── Data & Storage
│   ├── enterprise_documents/        # Sample documents
│   ├── faiss_index/                 # Vector store
│   ├── processed_data/              # Processed documents
│   └── Data/                        # Sample data files
├── Configuration
│   ├── requirements.txt             # Python dependencies
│   ├── run_enterprise_assistant.py  # Main launcher
│   └── config_examples.py           # Configuration examples
└── Documentation
    ├── README.md                    # This file
    └── document_metadata.json       # Document metadata
```

## Configuration

### Environment Variables

   ```bash
# Required for full functionality
OPENAI_API_KEY=your-openai-api-key-here

# Optional configurations
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-4
DATABASE_URL=sqlite:///enterprise_data.db
```

### Custom Configuration

Create a custom configuration file:

```python
# config_custom.py
CUSTOM_CONFIG = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "llm_model": "gpt-3.5-turbo",
    "database_url": "postgresql://user:pass@localhost/enterprise_db",
    "vector_store_path": "./custom_vector_store",
    "max_tokens": 2000,
    "temperature": 0.1
}
```

### Database Configuration

The system supports multiple database backends:

```python
# SQLite (default)
database_url = "sqlite:///enterprise_data.db"

# PostgreSQL
database_url = "postgresql://user:password@localhost/enterprise_db"

# SQL Server
database_url = "mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
```

## Use Cases

### 1. **Document Management**
- **Use Case**: Centralized knowledge base for company policies, procedures, and documentation
- **Benefits**: Quick access to relevant information, reduced search time, consistent answers
- **Example**: "What are the procedures for handling customer complaints?"

### 2. **Business Intelligence**
- **Use Case**: Data-driven decision making with natural language queries
- **Benefits**: Non-technical users can access complex data, real-time insights, automated reporting
- **Example**: "Show me sales performance by region for Q3"

### 3. **Customer Support**
- **Use Case**: Automated responses to common customer inquiries
- **Benefits**: Faster response times, 24/7 availability, consistent information
- **Example**: "How do I reset my password?"

### 4. **Compliance & Auditing**
- **Use Case**: Quick access to regulatory requirements and compliance procedures
- **Benefits**: Reduced compliance risk, faster audits, up-to-date information
- **Example**: "What are the data retention requirements for customer records?"

### 5. **Training & Onboarding**
- **Use Case**: Interactive learning system for new employees
- **Benefits**: Self-paced learning, consistent training, reduced training costs
- **Example**: "What are the safety procedures for warehouse operations?"

## Example Queries

### Document Queries
```
"What are the standard operating procedures for inventory management?"
"How do we handle supplier quality issues?"
"What are the temperature requirements for cold chain management?"
"How do we resolve customer delivery delays?"
"What are the key performance indicators for supply chain operations?"
```

### Data Queries
```
"What are the total sales by region?"
"Which product categories generate the most revenue?"
"Show me the top 10 products by sales volume"
"What is the customer lifetime value by segment?"
"Compare monthly sales trends for the last year"
```

### Integration Queries
```
"Based on our inventory procedures, what are the current stock levels?"
"Given our quality management policies, what are the defect rates by supplier?"
"Considering our delivery standards, what are the on-time delivery metrics?"
```

## Advanced Configuration

### Custom Embeddings

```python
from sentence_transformers import SentenceTransformer

# Use a different embedding model
custom_embeddings = SentenceTransformer('all-mpnet-base-v2')

# Configure RAG pipeline with custom embeddings
rag_pipeline = LangChainRAGPipeline(
    embedding_model=custom_embeddings,
    llm_model="gpt-4"
)
```

### Custom Prompts

```python
# Custom prompt template
custom_prompt = """
You are an expert enterprise consultant. Use the following context to provide detailed, actionable advice.

Context: {context}
Question: {question}

Provide a comprehensive answer with specific recommendations.
"""

# Use custom prompt in RAG pipeline
rag_pipeline.set_custom_prompt(custom_prompt)
```

### Database Views

```python
# Create custom database views
custom_views = {
    "monthly_performance": """
        SELECT 
            DATE_FORMAT(date, '%Y-%m') as month,
            SUM(sales_amount) as total_sales,
            COUNT(*) as transactions
        FROM sales 
        GROUP BY month
        ORDER BY month
    """,
    "customer_segments": """
        SELECT 
            customer_segment,
            COUNT(*) as customer_count,
            AVG(total_spent) as avg_spent
        FROM customers 
        GROUP BY customer_segment
    """
}

# Add views to SQL agent
sql_agent.add_custom_views(custom_views)
```

### Custom KPIs

```python
# Define custom KPIs
custom_kpis = {
    "inventory_turnover": {
        "description": "Inventory turnover ratio by product category",
        "query": """
            SELECT 
                p.category,
                SUM(s.quantity) / AVG(i.quantity) as turnover_ratio
            FROM sales s
            JOIN products p ON s.product_id = p.id
            JOIN inventory i ON p.id = i.product_id
            GROUP BY p.category
        """
    }
}

# Add custom KPIs to SQL agent
sql_agent.add_custom_kpis(custom_kpis)
```

## Monitoring & Analytics

### Usage Tracking

The system automatically tracks usage metrics:

   ```python
# Get usage analytics
analytics = rag_pipeline.get_usage_analytics()

print(f"Total queries: {analytics['total_queries']}")
print(f"Average response time: {analytics['avg_response_time']:.2f}s")
print(f"Most common queries: {analytics['top_queries']}")
```

### Performance Monitoring

```python
# Monitor system performance
status = rag_pipeline.get_system_status()

print(f"Vector store size: {status['vector_store_stats']['total_vectors']}")
print(f"Memory usage: {status['memory_usage']:.2f}MB")
print(f"Response time: {status['avg_response_time']:.2f}s")
```

### Custom Metrics

```python
# Track custom metrics
rag_pipeline.track_custom_metric("user_satisfaction", 4.5)
rag_pipeline.track_custom_metric("query_complexity", "high")

# Get custom metrics
custom_metrics = rag_pipeline.get_custom_metrics()
```

## Performance Optimization

### Memory Optimization

For large document collections:

```python
# Use memory-efficient settings
rag_pipeline = LangChainRAGPipeline(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=50,  # Less overlap
    max_tokens=1000,  # Limit response length
    use_gpu=False  # CPU-only for memory efficiency
)
```

### Database Optimization

```python
# Optimize database queries
sql_agent = EnterpriseSQLAgent(
    database_url="postgresql://...",
    connection_pool_size=10,
    query_timeout=30,
    enable_query_cache=True
)
```

### Caching

```python
# Enable response caching
rag_pipeline.enable_caching(
    cache_ttl=3600,  # Cache for 1 hour
    cache_size=1000  # Max 1000 cached responses
)
```

### Batch Processing

```python
# Process documents in batches
processor = EnterpriseKnowledgeProcessor()
processor.process_documents_batch(
    document_paths=document_files,
    batch_size=10,
    max_workers=4
)
```

## Security Considerations

### API Key Management

```python
# Secure API key handling
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file
api_key = os.getenv('OPENAI_API_KEY')

# Validate API key
if not api_key:
    raise ValueError("OpenAI API key is required")
```

### Input Validation

```python
# Validate user inputs
def validate_query(query: str) -> bool:
    if len(query) > 1000:
        return False
    if any(char in query for char in ['<', '>', 'script']):
        return False
    return True
```

### Rate Limiting

```python
# Implement rate limiting
from functools import wraps
import time

def rate_limit(max_requests=100, window=3600):
    def decorator(func):
        requests = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            requests[:] = [req for req in requests if now - req < window]
            
            if len(requests) >= max_requests:
                raise Exception("Rate limit exceeded")
            
            requests.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce chunk size and overlap
   - Use CPU-only mode
   - Process documents in smaller batches

2. **Slow Response Times**
   - Enable caching
   - Optimize database queries
   - Use smaller embedding models

3. **API Key Issues**
   - Verify API key is valid
   - Check rate limits
   - Ensure proper environment variable setup

4. **Database Connection Issues**
   - Verify database URL format
   - Check network connectivity
   - Ensure database permissions

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Debug RAG pipeline
rag_pipeline.enable_debug_mode()

# Debug SQL agent
sql_agent.enable_debug_mode()
```

## Future Enhancements

### Planned Features

1. **Multi-Modal Support**
   - Image and video processing
   - Audio transcription
   - Document OCR

2. **Advanced Analytics**
   - Predictive analytics
   - Anomaly detection
   - Trend analysis

3. **Integration Capabilities**
   - CRM integration
   - ERP system connectors
   - Third-party API support

4. **Enhanced Security**
   - Role-based access control
   - Data encryption
   - Audit logging

5. **Scalability Improvements**
   - Distributed processing
   - Load balancing
   - Auto-scaling

### Contributing

We welcome contributions! Please see our contributing guidelines for details.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Enterprise Knowledge Assistant** - Empowering organizations with intelligent document search and data analytics.
