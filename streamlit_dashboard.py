#!/usr/bin/env python3
"""
Streamlit Dashboard for Enterprise Knowledge Assistant

This module provides a comprehensive web interface for the Enterprise Knowledge
Management system, integrating both RAG-based document Q&A and SQL-based KPI queries.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local modules
try:
    from langchain_rag_pipeline import LangChainRAGPipeline
    from sql_agent import EnterpriseSQLAgent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
        padding: 0.5rem;
        background-color: #f0f8f0;
        border-left: 4px solid #2e8b57;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    .source-doc {
        background-color: #fff3cd;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
        font-size: 0.9rem;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
    """Main dashboard class for the Enterprise Knowledge Assistant"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.rag_pipeline = None
        self.sql_agent = None
        self._initialize_session_state()
        self._load_system_components()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = {}
        
        if 'last_query_result' not in st.session_state:
            st.session_state.last_query_result = None
        
        if 'selected_example_question' not in st.session_state:
            st.session_state.selected_example_question = ""
        
        if 'selected_sql_question' not in st.session_state:
            st.session_state.selected_sql_question = ""
        
        if 'selected_custom_sql' not in st.session_state:
            st.session_state.selected_custom_sql = ""
        
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    
    def _load_system_components(self):
        """Load RAG pipeline and SQL agent"""
        with st.spinner("Initializing system components..."):
            try:
                # Initialize RAG pipeline
                self.rag_pipeline = LangChainRAGPipeline(
                    openai_api_key=st.session_state.openai_api_key,
                    embedding_model="sentence-transformers"
                )
                
                # Initialize SQL agent
                self.sql_agent = EnterpriseSQLAgent(
                    openai_api_key=st.session_state.openai_api_key,
                    use_sqlite=True
                )
                
                # Update system status
                st.session_state.system_status = {
                    'rag_status': self.rag_pipeline.get_system_status(),
                    'sql_status': self.sql_agent.get_system_status(),
                    'last_updated': datetime.now().isoformat()
                }
                
            except Exception as e:
                st.error(f"Error initializing system components: {e}")
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<div class="main-header">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
        
        # System status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rag_status = st.session_state.system_status.get('rag_status', {})
            rag_ok = rag_status.get('vector_store_loaded', False)
            st.metric("RAG System", "Ready" if rag_ok else "Error")
        
        with col2:
            sql_status = st.session_state.system_status.get('sql_status', {})
            sql_ok = sql_status.get('database_connected', False)
            st.metric("Database", "Connected" if sql_ok else "Error")
        
        with col3:
            llm_status = rag_status.get('llm_available', False)
            st.metric("LLM", "Available" if llm_status else "Limited")
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("Configuration")
            
            # API Key input
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get('openai_api_key', ''),
                type='password',
                help="Enter your OpenAI API key for full functionality"
            )
            
            if api_key != st.session_state.get('openai_api_key', ''):
                st.session_state.openai_api_key = api_key
                st.rerun()
            
            # System status
            st.subheader("System Status")
            
            rag_status = st.session_state.system_status.get('rag_status', {})
            st.write(f"- Vector Store: {'Ready' if rag_status.get('vector_store_loaded') else 'Not Ready'}")
            st.write(f"- LLM: {'Available' if rag_status.get('llm_available') else 'Not Available'}")
            
            sql_status = st.session_state.system_status.get('sql_status', {})
            st.write(f"- Database: {'Connected' if sql_status.get('database_connected') else 'Not Connected'}")
            st.write(f"- Agent: {'Available' if sql_status.get('sql_agent_available') else 'Not Available'}")
            
            # Actions
            st.subheader("Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Chat History", use_container_width=True):
                    if hasattr(st.session_state, 'chat_history'):
                        st.session_state.chat_history = []
                    st.success("Chat history cleared!")
            
            with col2:
                if st.button("Show Available KPIs", use_container_width=True):
                    kpis = self.sql_agent.get_available_kpis()
                    st.write("Available KPIs:")
                    for name, desc in kpis.items():
                        st.write(f"- {name}: {desc}")
            
            if st.button("System Statistics", use_container_width=True):
                self._show_system_stats()
    
    def render_rag_interface(self):
        """Render the RAG document Q&A interface"""
        st.markdown('<div class="section-header">Document Q&A (RAG)</div>', unsafe_allow_html=True)
        
        # Question input
        default_value = st.session_state.selected_example_question
        user_question = st.text_input(
            "Ask a question about enterprise documents:",
            value=default_value,
            placeholder="e.g., What are the warehouse inventory management procedures?",
            key="rag_question"
        )
        if st.session_state.selected_example_question:
            st.session_state.selected_example_question = ""
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            use_conversation = st.checkbox("Conversation", value=False)
        with col2:
            show_sources = st.checkbox("Show Sources", value=True)
        
        # Example questions
        st.write("Example Questions:")
        example_questions = [
            "What are the standard procedures for warehouse inventory management?",
            "How should we handle supplier quality issues?",
            "What are the temperature requirements for cold chain management?",
            "How do we resolve customer delivery delays?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_rag_{i}", use_container_width=True):
                    st.session_state.selected_example_question = question
                    st.rerun()
        
        # Ask question button
        if user_question and st.button("Ask Question", type="primary", key="ask_rag"):
            self._process_rag_query(user_question, use_conversation, show_sources)
        
        # Display conversation history
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            st.subheader("Conversation History")
            for i, entry in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                with st.expander(f"Q{i+1}: {entry['question'][:50]}..."):
                    st.write(f"**Question:** {entry['question']}")
                    st.write(f"**Answer:** {entry['answer']}")
                    if entry.get('sources'):
                        st.write(f"**Sources:** {len(entry['sources'])} documents")
    
    def render_sql_interface(self):
        """Render the SQL agent interface for KPI queries"""
        st.markdown('<div class="section-header">Business Intelligence (SQL)</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["KPI Dashboard", "Natural Language Queries", "Custom SQL"])
        
        with tab1:
            self._render_kpi_dashboard()
        
        with tab2:
            self._render_natural_language_queries()
        
        with tab3:
            self._render_custom_sql()
    
    def _render_kpi_dashboard(self):
        """Render KPI dashboard"""
        st.write("Select a KPI to execute:")
        
        # Get available KPIs
        kpis = self.sql_agent.get_available_kpis()
        
        # KPI selection
        selected_kpi = st.selectbox(
            "Choose KPI:",
            options=list(kpis.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()}: {kpis[x]}"
        )
        
        if st.button("Execute KPI", type="primary", use_container_width=True):
            self._execute_kpi_query(selected_kpi)
        
        # Quick KPI buttons
        st.write("Quick Actions:")
        kpi_list = list(kpis.keys())
        cols = st.columns(3)
        
        for i, kpi in enumerate(kpi_list[:6]):  # Show first 6 KPIs
            with cols[i % 3]:
                if st.button(f"{kpi.replace('_', ' ').title()}", key=f"quick_kpi_{i}", use_container_width=True):
                    self._execute_kpi_query(kpi)
    
    def _render_natural_language_queries(self):
        """Render natural language SQL interface"""
        st.write("Ask questions about your business data in natural language:")
        
        # Question input
        default_value = st.session_state.selected_sql_question
        user_question = st.text_input(
            "Ask a question about your data:",
            value=default_value,
            placeholder="e.g., What are the total sales by region?",
            key="nl_sql_question"
        )
        if st.session_state.selected_sql_question:
            st.session_state.selected_sql_question = ""
        
        # Example questions
        st.write("Example Questions:")
        example_questions = [
            "What are the total sales by region?",
            "Which product categories generate the most revenue?",
            "Show me the top 5 products by sales",
            "What is the customer lifetime value by segment?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_nl_{i}", use_container_width=True):
                    st.session_state.selected_sql_question = question
                    st.rerun()
        
        # Execute button
        if user_question and st.button("Ask Question", type="primary", key="ask_nl_sql"):
            self._process_nl_sql_query(user_question)
    
    def _render_custom_sql(self):
        """Render custom SQL query interface"""
        st.write("Execute custom SQL queries directly:")
        
        # SQL input
        default_value = st.session_state.selected_custom_sql
        sql_query = st.text_area(
            "Enter SQL query:",
            value=default_value,
            height=100,
            placeholder="SELECT * FROM sales LIMIT 10",
            key="custom_sql"
        )
        if st.session_state.selected_custom_sql:
            st.session_state.selected_custom_sql = ""
        
        if st.button("Execute", type="primary", use_container_width=True):
            self._execute_custom_sql(sql_query)
        
        if st.button("Show Schema", use_container_width=True):
            self._show_database_schema()
        
        # Example queries
        st.write("Example Queries:")
        example_queries = {
            "Top Products": "SELECT ProductName, SUM(SalesAmount) as TotalSales FROM sales s JOIN product p ON s.ProductKey = p.ProductKey GROUP BY ProductName ORDER BY TotalSales DESC LIMIT 10",
            "Sales by Month": "SELECT DATE_FORMAT(Date, '%Y-%m') as Month, SUM(SalesAmount) as TotalSales FROM sales s JOIN date d ON s.DateKey = d.DateKey GROUP BY Month ORDER BY Month",
            "Customer Count": "SELECT COUNT(DISTINCT CustomerKey) as TotalCustomers FROM sales"
        }
        
        for name, query in example_queries.items():
            if st.button(name, key=f"example_sql_{name}", use_container_width=True):
                st.session_state.selected_custom_sql = query
                st.rerun()
    
    def _process_rag_query(self, question: str, use_conversation: bool = False, show_sources: bool = True):
        """Process a RAG query and display results"""
        if not self.rag_pipeline:
            st.error("RAG pipeline not available")
            return
        
        with st.spinner("Processing your question..."):
            try:
                result = self.rag_pipeline.query(question, use_conversation=use_conversation)
                
                # Add to chat history
                chat_entry = {
                    'type': 'rag',
                    'question': question,
                    'answer': result['answer'],
                    'sources': result.get('sources', []),
                    'timestamp': result['timestamp'],
                    'model': result.get('model_used', 'unknown')
                }
                st.session_state.chat_history.append(chat_entry)
                
                # Display result
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {result['answer']}")
                
                # Display sources
                if result.get('sources') and show_sources:
                    st.markdown("**Sources:**")
                    for i, source in enumerate(result['sources'][:3], 1):
                        st.markdown(f"- {source['filename']} ({source['document_type']})")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", len(result.get('sources', [])))
                with col2:
                    st.metric("Model", result.get('model_used', 'N/A'))
                with col3:
                    st.metric("Response Time", "< 5s")  # Approximate
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
    
    def _execute_kpi(self, kpi_name: str):
        """Execute a KPI query"""
        if not self.sql_agent:
            st.error("SQL agent not available")
            return
        
        with st.spinner(f"Executing KPI: {kpi_name}..."):
            try:
                result = self.sql_agent.execute_kpi_query(kpi_name)
                
                if 'error' in result:
                    st.error(f"Error executing KPI: {result['error']}")
                else:
                    st.session_state.last_query_result = {
                        'type': 'kpi',
                        'data': result
                    }
                    st.success(f"KPI '{kpi_name}' executed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error executing KPI: {e}")
    
    def _process_nl_sql_query(self, question: str):
        """Process a natural language SQL query"""
        if not self.sql_agent:
            st.error("SQL agent not available")
            return
        
        with st.spinner("Processing your business question..."):
            try:
                result = self.sql_agent.query_with_agent(question)
                
                # Display result
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {result['answer']}")
                
                # If there's KPI data, display it
                if 'kpi_result' in result and result['kpi_result'].get('data'):
                    self._display_kpi_results(result['kpi_result'])
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to chat history
                chat_entry = {
                    'type': 'sql',
                    'question': question,
                    'answer': result['answer'],
                    'agent_used': result.get('agent_used', False),
                    'timestamp': result['timestamp']
                }
                st.session_state.chat_history.append(chat_entry)
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
    
    def _execute_custom_sql(self, sql_query: str):
        """Execute a custom SQL query"""
        if not self.sql_agent:
            st.error("SQL agent not available")
            return
        
        with st.spinner("Executing SQL query..."):
            try:
                result = self.sql_agent.execute_custom_query(sql_query)
                
                if 'error' in result:
                    st.error(f"SQL Error: {result['error']}")
                else:
                    st.success(f"Query executed successfully! {result['row_count']} rows returned.")
                    
                    # Display results
                    if result['data']:
                        df = pd.DataFrame(result['data'])
                        st.dataframe(df, use_container_width=True)
                        
                        # Show query info
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rows", result['row_count'])
                        with col2:
                            st.metric("Columns", len(result['columns']))
                    else:
                        st.info("Query executed successfully but returned no data.")
                        
            except Exception as e:
                st.error(f"Error executing query: {e}")
    
    def _display_kpi_results(self, kpi_result: Dict[str, Any]):
        """Display KPI results with visualizations"""
        if 'error' in kpi_result:
            st.error(f"KPI Error: {kpi_result['error']}")
            return
        
        if not kpi_result.get('data'):
            st.warning("No data returned from KPI query")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(kpi_result['data'])
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records", kpi_result['row_count'])
        with col2:
            st.metric("Columns", len(kpi_result['columns']))
        with col3:
            st.metric("KPI", kpi_result['kpi_name'].replace('_', ' ').title())
        
        # Display data table
        st.subheader("Data")
        st.dataframe(df, use_container_width=True)
        
        # Create visualizations based on KPI type
        if kpi_result['kpi_name'] == 'sales_by_region' and 'TotalSales' in df.columns:
            fig = px.bar(df, x='Region', y='TotalSales', title='Total Sales by Region')
            st.plotly_chart(fig, use_container_width=True)
            
        elif kpi_result['kpi_name'] == 'revenue_by_product_category' and 'TotalRevenue' in df.columns:
            fig = px.pie(df, values='TotalRevenue', names='Category', title='Revenue by Product Category')
            st.plotly_chart(fig, use_container_width=True)
            
        elif kpi_result['kpi_name'] == 'monthly_sales_trend' and 'MonthlySales' in df.columns:
            fig = px.line(df, x='MonthName', y='MonthlySales', title='Monthly Sales Trend')
            st.plotly_chart(fig, use_container_width=True)
            
        elif kpi_result['kpi_name'] == 'top_products' and 'TotalRevenue' in df.columns:
            fig = px.bar(df, x='ProductName', y='TotalRevenue', title='Top Products by Revenue')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show raw query
        with st.expander("View SQL Query"):
            st.code(kpi_result.get('query', ''), language='sql')
    
    def _show_database_schema(self):
        """Display database schema information"""
        if not self.sql_agent:
            st.error("SQL agent not available")
            return
        
        try:
            schema = self.sql_agent.get_database_schema()
            
            if 'error' in schema:
                st.error(f"Error getting schema: {schema['error']}")
                return
            
            st.subheader("Database Schema")
            
            for table_name, table_info in schema['tables'].items():
                with st.expander(f"{table_name} ({table_info['column_count']} columns)"):
                    st.write(f"**Description:** {table_info.get('description', 'No description available')}")
                    
                    if table_info.get('columns'):
                        st.write("**Columns:**")
                        for col_name, col_info in table_info['columns'].items():
                            st.write(f"- {col_name}: {col_info['type']} ({col_info.get('description', 'No description')})")
            
            # Show views if available
            if schema.get('views'):
                st.subheader("Database Views")
                for view_name, view_info in schema['views'].items():
                    with st.expander(f"{view_name} ({view_info['column_count']} columns)"):
                        st.write(f"**Description:** {view_info.get('description', 'No description available')}")
                        
                        if view_info.get('columns'):
                            st.write("**Columns:**")
                            for col_name, col_info in view_info['columns'].items():
                                st.write(f"- {col_name}: {col_info['type']}")
        
        except Exception as e:
            st.error(f"Error displaying schema: {e}")
    
    def render_main_interface(self):
        """Render the main interface with tabs"""
        tab1, tab2, tab3 = st.tabs(["Dashboard", "Document Q&A", "Business Intelligence"])
        
        with tab1:
            self._render_dashboard_overview()
        
        with tab2:
            self.render_rag_interface()
        
        with tab3:
            self.render_sql_interface()
    
    def _render_dashboard_overview(self):
        """Render the main dashboard overview"""
        st.markdown('<div class="main-header">Enterprise Knowledge Assistant Dashboard</div>', unsafe_allow_html=True)
        
        # Overview
        st.write("Welcome to the Enterprise Knowledge Assistant! This system provides:")
        st.write("- Document Q&A: Ask questions about your enterprise documents using AI")
        st.write("- Business Intelligence: Query your data and get insights with natural language")
        st.write("- Semantic Search: Find relevant information across all your knowledge base")
        st.write("- KPI Dashboard: Monitor key performance indicators and metrics")
    
    def _show_system_stats(self):
        """Show system statistics"""
        if not self.rag_pipeline:
            st.error("RAG pipeline not available")
            return
        
        try:
            stats = self.rag_pipeline.knowledge_processor.get_document_statistics()
            st.write("System Statistics:")
            st.json(stats)
        except Exception as e:
            st.error(f"Error getting system stats: {e}")
    
    def run(self):
        """Run the Streamlit dashboard"""
        # Render header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main interface
        self.render_main_interface()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "Enterprise Knowledge Assistant | "
            "Powered by LangChain • OpenAI • FAISS • Streamlit"
        )


def main():
    """Main function to run the dashboard"""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 