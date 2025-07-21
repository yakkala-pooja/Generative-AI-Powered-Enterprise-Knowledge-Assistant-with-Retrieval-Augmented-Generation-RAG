#!/usr/bin/env python3
"""
FastAPI Backend for Enterprise Knowledge Assistant

This module provides REST API endpoints for the Enterprise Knowledge Management system,
exposing both RAG-based document Q&A and SQL-based KPI query functionalities.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import asyncio
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Local imports
try:
    from langchain_rag_pipeline import LangChainRAGPipeline
    from sql_agent import EnterpriseSQLAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class RAGQuery(BaseModel):
    question: str = Field(..., description="Question to ask about the documents")
    use_conversation: bool = Field(False, description="Whether to use conversational memory")
    max_sources: int = Field(5, description="Maximum number of source documents to return")

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    timestamp: str
    model_used: str
    source_count: int
    conversation_used: bool = False

class SQLQuery(BaseModel):
    question: str = Field(..., description="Natural language question about the data")
    use_agent: bool = Field(True, description="Whether to use the SQL agent for query processing")

class SQLResponse(BaseModel):
    question: str
    answer: str
    data: Optional[List[Dict[str, Any]]] = None
    agent_used: bool
    timestamp: str
    query_type: str

class KPIRequest(BaseModel):
    kpi_name: str = Field(..., description="Name of the KPI to execute")

class KPIResponse(BaseModel):
    kpi_name: str
    description: str
    data: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time: str
    query: str

class CustomSQLRequest(BaseModel):
    sql_query: str = Field(..., description="SQL query to execute")
    limit: Optional[int] = Field(1000, description="Maximum number of rows to return")

class CustomSQLResponse(BaseModel):
    sql_query: str
    data: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    execution_time: str

class SystemStatus(BaseModel):
    rag_status: Dict[str, Any]
    sql_status: Dict[str, Any]
    api_version: str
    timestamp: str
    health: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str
    endpoint: str

# Global variables for system components
rag_pipeline: Optional[LangChainRAGPipeline] = None
sql_agent: Optional[EnterpriseSQLAgent] = None

# Security
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Enterprise Knowledge Assistant API...")
    await initialize_system_components()
    yield
    # Shutdown
    logger.info("Shutting down Enterprise Knowledge Assistant API...")

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="REST API for RAG-based document Q&A and SQL-based business intelligence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_system_components():
    """Initialize RAG pipeline and SQL agent"""
    global rag_pipeline, sql_agent
    
    try:
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = LangChainRAGPipeline(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            embedding_model="sentence-transformers"
        )
        logger.info("RAG pipeline initialized successfully")
        
        logger.info("Initializing SQL agent...")
        sql_agent = EnterpriseSQLAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            use_sqlite=True
        )
        logger.info("SQL agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing system components: {e}")
        # Don't fail startup, allow limited functionality

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication dependency (extend as needed)"""
    # For now, just return a simple user dict
    # In production, validate the token and return user info
    return {"user_id": "demo_user", "permissions": ["read", "write"]}

# Health check endpoint
@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# System status endpoint
@app.get("/api/v1/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        rag_status = rag_pipeline.get_system_status() if rag_pipeline else {"error": "RAG pipeline not available"}
        sql_status = sql_agent.get_system_status() if sql_agent else {"error": "SQL agent not available"}
        
        return SystemStatus(
            rag_status=rag_status,
            sql_status=sql_status,
            api_version="1.0.0",
            timestamp=datetime.now().isoformat(),
            health="healthy" if rag_pipeline and sql_agent else "limited"
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG Endpoints
@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_documents(
    query: RAGQuery,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Query documents using RAG pipeline"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        logger.info(f"Processing RAG query from user {user['user_id']}: {query.question}")
        
        # Process the query
        result = rag_pipeline.query(
            question=query.question,
            use_conversation=query.use_conversation
        )
        
        # Limit sources if requested
        sources = result.get("sources", [])[:query.max_sources]
        
        # Log query for analytics (background task)
        background_tasks.add_task(
            log_query_analytics,
            "rag",
            query.question,
            user["user_id"],
            len(sources)
        )
        
        return RAGResponse(
            answer=result["answer"],
            sources=sources,
            query=result["query"],
            timestamp=result["timestamp"],
            model_used=result["model_used"],
            source_count=len(sources),
            conversation_used=query.use_conversation
        )
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/rag/conversation-history")
async def get_conversation_history(user: dict = Depends(get_current_user)):
    """Get conversation history for the current user"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        history = rag_pipeline.get_conversation_history()
        return {"conversation_history": history, "count": len(history)}
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/rag/conversation-history")
async def clear_conversation_history(user: dict = Depends(get_current_user)):
    """Clear conversation history for the current user"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        rag_pipeline.clear_conversation_history()
        return {"message": "Conversation history cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# SQL Agent Endpoints
@app.post("/api/v1/sql/query", response_model=SQLResponse)
async def query_data_nl(
    query: SQLQuery,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Query data using natural language"""
    if not sql_agent:
        raise HTTPException(status_code=503, detail="SQL agent not available")
    
    try:
        logger.info(f"Processing SQL query from user {user['user_id']}: {query.question}")
        
        # Process the query
        if query.use_agent:
            result = sql_agent.query_with_agent(query.question)
        else:
            # Fallback to direct query mapping
            result = sql_agent._fallback_direct_query(query.question)
        
        # Extract data if available
        data = None
        if "kpi_result" in result and result["kpi_result"].get("data"):
            data = result["kpi_result"]["data"]
        
        # Log query for analytics
        background_tasks.add_task(
            log_query_analytics,
            "sql",
            query.question,
            user["user_id"],
            len(data) if data else 0
        )
        
        return SQLResponse(
            question=result["question"],
            answer=result["answer"],
            data=data,
            agent_used=result.get("agent_used", False),
            timestamp=result["timestamp"],
            query_type="natural_language"
        )
        
    except Exception as e:
        logger.error(f"Error processing SQL query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sql/kpis")
async def get_available_kpis():
    """Get list of available KPIs"""
    if not sql_agent:
        raise HTTPException(status_code=503, detail="SQL agent not available")
    
    try:
        kpis = sql_agent.get_available_kpis()
        return {
            "kpis": kpis,
            "count": len(kpis),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sql/kpi", response_model=KPIResponse)
async def execute_kpi(
    kpi_request: KPIRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Execute a specific KPI query"""
    if not sql_agent:
        raise HTTPException(status_code=503, detail="SQL agent not available")
    
    try:
        logger.info(f"Executing KPI {kpi_request.kpi_name} for user {user['user_id']}")
        
        result = sql_agent.execute_kpi_query(kpi_request.kpi_name)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Log KPI execution
        background_tasks.add_task(
            log_query_analytics,
            "kpi",
            kpi_request.kpi_name,
            user["user_id"],
            result["row_count"]
        )
        
        return KPIResponse(
            kpi_name=result["kpi_name"],
            description=result["description"],
            data=result["data"],
            row_count=result["row_count"],
            columns=result["columns"],
            execution_time=result["execution_time"],
            query=result["query"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing KPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/sql/custom", response_model=CustomSQLResponse)
async def execute_custom_sql(
    sql_request: CustomSQLRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Execute a custom SQL query"""
    if not sql_agent:
        raise HTTPException(status_code=503, detail="SQL agent not available")
    
    try:
        logger.info(f"Executing custom SQL for user {user['user_id']}")
        
        # Add LIMIT clause if not present and limit is specified
        sql_query = sql_request.sql_query
        if sql_request.limit and "LIMIT" not in sql_query.upper():
            sql_query += f" LIMIT {sql_request.limit}"
        
        result = sql_agent.execute_custom_query(sql_query)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Log custom query execution
        background_tasks.add_task(
            log_query_analytics,
            "custom_sql",
            "custom_query",
            user["user_id"],
            result["row_count"]
        )
        
        return CustomSQLResponse(
            sql_query=result["sql_query"],
            data=result["data"],
            row_count=result["row_count"],
            columns=result["columns"],
            execution_time=result["execution_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing custom SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/sql/schema")
async def get_database_schema():
    """Get database schema information"""
    if not sql_agent:
        raise HTTPException(status_code=503, detail="SQL agent not available")
    
    try:
        schema = sql_agent.get_database_schema()
        
        if "error" in schema:
            raise HTTPException(status_code=500, detail=schema["error"])
        
        return schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Monitoring Endpoints
@app.get("/api/v1/analytics/usage")
async def get_usage_analytics(
    days: int = Query(7, description="Number of days to include in analytics"),
    user: dict = Depends(get_current_user)
):
    """Get usage analytics (simplified implementation)"""
    try:
        # In a real implementation, this would query a analytics database
        # For now, return mock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        mock_analytics = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "rag_queries": {
                "total": 45,
                "avg_per_day": 6.4,
                "top_topics": ["inventory management", "supplier qualification", "quality control"]
            },
            "sql_queries": {
                "total": 32,
                "avg_per_day": 4.6,
                "kpi_executions": 18,
                "custom_queries": 14
            },
            "system_health": {
                "uptime_percentage": 99.8,
                "avg_response_time_ms": 1250,
                "error_rate_percentage": 0.2
            }
        }
        
        return mock_analytics
        
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.post("/api/v1/documents/add")
async def add_documents(
    file_paths: List[str] = Body(..., description="List of document file paths to add"),
    user: dict = Depends(get_current_user)
):
    """Add new documents to the knowledge base"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        logger.info(f"Adding {len(file_paths)} documents for user {user['user_id']}")
        
        result = rag_pipeline.add_documents(file_paths)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "message": "Documents added successfully",
            "documents_added": result["documents_added"],
            "chunks_added": result["chunks_added"],
            "total_documents": result["total_documents"],
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/documents/statistics")
async def get_document_statistics():
    """Get document processing statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        stats = rag_pipeline.knowledge_processor.get_document_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting document statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def log_query_analytics(query_type: str, query: str, user_id: str, result_count: int):
    """Log query for analytics (background task)"""
    try:
        # In a real implementation, this would write to an analytics database
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "query": query,
            "user_id": user_id,
            "result_count": result_count
        }
        
        # For now, just log to file
        log_file = "query_analytics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging query analytics: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested endpoint was not found",
            "timestamp": datetime.now().isoformat(),
            "endpoint": str(request.url)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat(),
            "endpoint": str(request.url)
        }
    )

# Development server configuration
def run_development_server():
    """Run the development server"""
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    print("Starting Enterprise Knowledge Assistant API Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("ReDoc Documentation: http://localhost:8000/redoc")
    run_development_server() 