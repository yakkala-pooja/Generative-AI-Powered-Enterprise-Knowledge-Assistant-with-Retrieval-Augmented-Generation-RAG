# 🏢 Enterprise Knowledge Assistant with Retrieval-Augmented Generation (RAG)

A comprehensive enterprise knowledge management system that combines document-based question answering with structured database queries using state-of-the-art AI technologies.

## 🌟 Features

### 📚 Document Q&A (RAG Pipeline)
- **Semantic Document Search**: Query enterprise documents using natural language
- **LangChain Integration**: Powered by LangChain for robust RAG implementation
- **GPT-4 Integration**: Intelligent response generation using OpenAI GPT-4
- **FAISS Vector Store**: Efficient similarity search with FAISS indexing
- **Multiple Embedding Models**: Support for OpenAI embeddings and sentence-transformers
- **Conversational Memory**: Multi-turn conversations with context retention

### 📊 Business Intelligence (SQL Agent)
- **Natural Language to SQL**: Query databases using plain English
- **Predefined KPIs**: Ready-to-use key performance indicators
- **Custom SQL Queries**: Execute complex SQL queries through the interface
- **Multiple Database Support**: SQLite, PostgreSQL, SQL Server compatibility
- **Automated Insights**: AI-powered data analysis and recommendations

### 🌐 Multi-Interface Access
- **Streamlit Dashboard**: Interactive web interface with rich visualizations
- **FastAPI Backend**: RESTful API endpoints for integration
- **Command Line Interface**: Direct terminal access for quick queries
- **Comprehensive Demo**: Interactive demonstration of all features

### 🔧 Enterprise-Ready Features
- **Document Processing**: Automated ingestion and chunking of enterprise documents
- **Security**: Authentication and access control mechanisms
- **Analytics**: Usage tracking and performance monitoring
- **Scalability**: Designed for enterprise-scale deployments
- **Extensibility**: Modular architecture for easy customization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional but recommended for full functionality)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Generative-AI-Powered-Enterprise-Knowledge-Assistant-with-Retrieval-Augmented-Generation-RAG-
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key** (optional):
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Quick demo**:
```bash
python run_enterprise_assistant.py demo
```

### 🎯 Usage Options

#### 1. **Interactive Demo** (Recommended for first-time users)
```bash
python run_enterprise_assistant.py demo
```
Runs a comprehensive demonstration showcasing all features.

#### 2. **Web Dashboard** (Best for regular use)
```bash
python run_enterprise_assistant.py streamlit
```
Access at: http://localhost:8501

#### 3. **API Server** (For integration)
```bash
python run_enterprise_assistant.py api
```
- API Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### 4. **Interactive CLI**
```bash
python run_enterprise_assistant.py interactive
```

#### 5. **All Services**
```bash
python run_enterprise_assistant.py all
```
Starts both Streamlit dashboard and FastAPI server.

## 📁 Project Structure

```
├── 📋 Core Components
│   ├── langchain_rag_pipeline.py      # LangChain RAG implementation
│   ├── sql_agent.py                   # SQL agent for database queries
│   ├── enterprise_knowledge_processor.py # Document processing engine
│   └── contoso_abf_loader.py          # Sample data loader
│
├── 🌐 Interfaces
│   ├── streamlit_dashboard.py         # Web dashboard interface
│   ├── fastapi_backend.py             # REST API backend
│   └── enterprise_assistant_demo.py   # Comprehensive demo script
│
├── 📊 Data & Storage
│   ├── enterprise_documents/          # Sample enterprise documents
│   ├── faiss_index/                   # FAISS vector store
│   ├── processed_data/                # Processed document cache
│   └── Data/                          # Sample datasets
│
├── ⚙️ Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── config_examples.py             # Configuration examples
│   └── run_enterprise_assistant.py    # Main launcher script
│
└── 📚 Documentation
    ├── README.md                      # This file
    └── document_metadata.json         # Document processing metadata
```

## 🔧 Configuration

### Environment Variables
```bash
# Required for full LLM functionality
export OPENAI_API_KEY="your-openai-api-key"

# Optional database configuration
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"

# Optional API configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"
```

### System Requirements
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space for vector indices
- **Network**: Internet access for OpenAI API (if used)

## 📖 API Documentation

### RAG Endpoints
- `POST /api/v1/rag/query` - Query documents with natural language
- `GET /api/v1/rag/conversation-history` - Get conversation history
- `DELETE /api/v1/rag/conversation-history` - Clear conversation history

### SQL Agent Endpoints
- `POST /api/v1/sql/query` - Natural language data queries
- `GET /api/v1/sql/kpis` - List available KPIs
- `POST /api/v1/sql/kpi` - Execute specific KPI
- `POST /api/v1/sql/custom` - Execute custom SQL queries
- `GET /api/v1/sql/schema` - Get database schema information

### System Endpoints
- `GET /health` - Health check
- `GET /api/v1/status` - Comprehensive system status
- `GET /api/v1/analytics/usage` - Usage analytics

## 🎯 Use Cases

### 1. **Supply Chain Management**
- Query SOPs and procedures
- Analyze performance metrics
- Track supplier compliance
- Monitor inventory levels

### 2. **Quality Management**
- Access quality procedures
- Investigate defect patterns
- Monitor compliance metrics
- Analyze customer feedback

### 3. **Business Intelligence**
- Generate performance reports
- Analyze sales trends
- Monitor KPIs
- Create executive dashboards

### 4. **Knowledge Management**
- Search enterprise documentation
- Get procedural guidance
- Access historical reports
- Find relevant case studies

## 🔍 Example Queries

### Document Q&A (RAG)
```
"What are the standard procedures for warehouse inventory management?"
"How should we handle supplier quality issues?"
"What are the temperature requirements for cold chain management?"
"How do we resolve customer delivery delays?"
```

### Business Intelligence (SQL)
```
"What are the total sales by region?"
"Which product categories generate the most revenue?"
"Show me the top 10 customers by lifetime value"
"What is the monthly sales trend this year?"
```

## 🛠️ Advanced Configuration

### Custom Document Types
Add new document types by updating the `document_type` classification in `enterprise_knowledge_processor.py`:

```python
def _classify_document_type(self, filename: str) -> str:
    if 'policy' in filename.lower():
        return 'policy_document'
    # Add more classifications
```

### Custom KPIs
Add new KPIs in `sql_agent.py`:

```python
def _define_kpis(self) -> Dict[str, Dict[str, str]]:
    return {
        "your_custom_kpi": {
            "description": "Description of your KPI",
            "query": "SELECT ... FROM ..."
        }
    }
```

### Database Integration
For production databases, update the connection string:

```python
# PostgreSQL
DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"

# SQL Server
DATABASE_URL = "mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"
```

## 🔐 Security Considerations

### Production Deployment
- Set proper CORS origins in `fastapi_backend.py`
- Implement proper authentication and authorization
- Use environment variables for sensitive configuration
- Enable HTTPS for web interfaces
- Regularly update dependencies

### API Security
- Implement rate limiting
- Add input validation and sanitization
- Use API keys or JWT tokens for authentication
- Monitor and log API usage

## 📊 Monitoring & Analytics

### Built-in Analytics
- Query tracking and analysis
- System performance monitoring
- Usage pattern identification
- Error rate monitoring

### Custom Analytics
Extend the analytics by implementing custom logging in `fastapi_backend.py`:

```python
async def log_query_analytics(query_type: str, query: str, user_id: str, result_count: int):
    # Your custom analytics logic
    pass
```

## 🚀 Performance Optimization

### Vector Store Optimization
- Use appropriate chunk sizes (default: 1000 characters)
- Optimize embedding model selection
- Consider GPU acceleration for large datasets

### Database Optimization
- Create appropriate indexes
- Use database views for complex queries
- Implement query caching where appropriate

### Scaling Considerations
- Use Redis for session management
- Implement horizontal scaling with load balancers
- Consider cloud-based vector stores for large deployments

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests and demos
5. Submit pull requests

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document functions and classes
- Write comprehensive tests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**2. OpenAI API Issues**
```bash
# Verify API key
echo $OPENAI_API_KEY

# Check API quota and usage
```

**3. Vector Store Issues**
```bash
# Regenerate vector store
python enterprise_knowledge_processor.py
```

**4. Database Connection Issues**
```bash
# Check database URL and credentials
# Verify database server is running
```

### Getting Help
- Check the demo output for system status
- Review logs for detailed error messages
- Ensure all dependencies are properly installed
- Verify API keys and database connections

### Performance Issues
- Monitor memory usage during vector operations
- Check disk space for vector store files
- Optimize query complexity for better response times

## 🔮 Future Enhancements

### Planned Features
- Multi-language document support
- Advanced analytics dashboard
- Integration with more vector databases
- Enhanced security features
- Mobile-responsive interface improvements

### Integration Opportunities
- Microsoft Office 365 integration
- Slack/Teams bot integration
- Enterprise SSO integration
- Cloud storage connectors (S3, Azure Blob, etc.)

---

**🏢 Enterprise Knowledge Assistant** - Empowering organizations with intelligent document search and data analytics.

For more information, examples, and advanced configuration, run the comprehensive demo:
```bash
python run_enterprise_assistant.py demo
```
