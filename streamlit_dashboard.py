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
    /* Modern color scheme and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Professional header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        position: relative;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    /* Enhanced section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1.5rem 0 1rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Modern metric containers */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 0.75rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Enhanced chat messages */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 16px;
        border-left: 5px solid #667eea;
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Professional source documents */
    .source-doc {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        border: 1px solid #fdba74;
        font-size: 0.9rem;
        font-weight: 500;
        color: #92400e;
        transition: all 0.3s ease;
    }
    
    .source-doc:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(251, 146, 60, 0.3);
    }
    
    /* Modern status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-ready { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Professional cards */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Modern sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    /* Enhanced text inputs */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Professional error and success messages */
    .error-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #991b1b;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #fecaca;
        margin: 1rem 0;
        border-left: 5px solid #ef4444;
    }
    
    .success-message {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #166534;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
        margin: 1rem 0;
        border-left: 5px solid #10b981;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
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
        try:
            # Initialize RAG pipeline
            if not hasattr(self, 'rag_pipeline') or self.rag_pipeline is None:
                self.rag_pipeline = LangChainRAGPipeline(
                    openai_api_key=st.session_state.openai_api_key,
                    embedding_model="sentence-transformers"
                )
            
            # Initialize SQL agent
            if not hasattr(self, 'sql_agent') or self.sql_agent is None:
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
            # Set default status if initialization fails
            st.session_state.system_status = {
                'rag_status': {'vector_store_loaded': False, 'llm_available': False},
                'sql_status': {'database_connected': False, 'sql_agent_available': False},
                'last_updated': datetime.now().isoformat()
            }
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                <div style="font-size: 3rem;">🤖</div>
                <div>
                    <div style="font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem;">
                        Enterprise Knowledge Assistant
                    </div>
                    <div style="font-size: 1.2rem; font-weight: 400; color: #64748b; margin-top: 0.5rem;">
                        AI-Powered Document Intelligence & Business Analytics
                    </div>
                </div>
                <div style="font-size: 3rem;">📊</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a subtle separator
        st.markdown("---")
        
        # Quick status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rag_status = st.session_state.system_status.get('rag_status', {})
            status_color = "#10b981" if rag_status.get('vector_store_loaded') else "#f59e0b"
            st.markdown(f"""
            <div class="metric-container">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div class="status-indicator status-ready" style="background-color: {status_color};"></div>
                    <strong>RAG System</strong>
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                    {'Ready' if rag_status.get('vector_store_loaded') else 'Loading'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sql_status = st.session_state.system_status.get('sql_status', {})
            status_color = "#10b981" if sql_status.get('database_connected') else "#f59e0b"
            st.markdown(f"""
            <div class="metric-container">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div class="status-indicator status-ready" style="background-color: {status_color};"></div>
                    <strong>Database</strong>
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                    {'Connected' if sql_status.get('database_connected') else 'Disconnected'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            llm_status = st.session_state.system_status.get('rag_status', {})
            status_color = "#10b981" if llm_status.get('llm_available') else "#f59e0b"
            st.markdown(f"""
            <div class="metric-container">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div class="status-indicator status-ready" style="background-color: {status_color};"></div>
                    <strong>AI Model</strong>
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {status_color};">
                    {'Available' if llm_status.get('llm_available') else 'Limited'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Show current time
            current_time = datetime.now().strftime("%H:%M")
            st.markdown(f"""
            <div class="metric-container">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 1.2rem;">🕒</div>
                    <strong>Last Updated</strong>
                </div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #667eea;">
                    {current_time}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #e2e8f0; margin-bottom: 2rem;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #2d3748; margin-bottom: 0.5rem;">
                    ⚙️ Configuration
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">
                    System Settings & API Keys
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # API Key Configuration
            st.markdown("""
            <div class="info-card">
                <div style="font-weight: 600; color: #2d3748; margin-bottom: 1rem;">
                    🔑 OpenAI API Key
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get('openai_api_key', ''),
                type='password',
                help="Enter your OpenAI API key for full functionality",
                placeholder="sk-..."
            )
            
            if api_key != st.session_state.get('openai_api_key', ''):
                st.session_state.openai_api_key = api_key
                # Reinitialize components with new API key
                self.rag_pipeline = None
                self.sql_agent = None
                self._load_system_components()
            
            # System Status
            st.markdown("""
            <div class="info-card">
                <div style="font-weight: 600; color: #2d3748; margin-bottom: 1rem;">
                    📊 System Status
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            rag_status = st.session_state.system_status.get('rag_status', {})
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div class="status-indicator status-ready" style="background-color: {'#10b981' if rag_status.get('vector_store_loaded') else '#f59e0b'};"></div>
                <span style="font-size: 0.9rem;">Vector Store: {'Ready' if rag_status.get('vector_store_loaded') else 'Not Ready'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div class="status-indicator status-ready" style="background-color: {'#10b981' if rag_status.get('llm_available') else '#f59e0b'};"></div>
                <span style="font-size: 0.9rem;">LLM: {'Available' if rag_status.get('llm_available') else 'Not Available'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            sql_status = st.session_state.system_status.get('sql_status', {})
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div class="status-indicator status-ready" style="background-color: {'#10b981' if sql_status.get('database_connected') else '#f59e0b'};"></div>
                <span style="font-size: 0.9rem;">Database: {'Connected' if sql_status.get('database_connected') else 'Not Connected'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div class="status-indicator status-ready" style="background-color: {'#10b981' if sql_status.get('sql_agent_available') else '#f59e0b'};"></div>
                <span style="font-size: 0.9rem;">SQL Agent: {'Available' if sql_status.get('sql_agent_available') else 'Not Available'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Actions
            st.markdown("""
            <div class="info-card">
                <div style="font-weight: 600; color: #2d3748; margin-bottom: 1rem;">
                    🛠️ Actions
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    if hasattr(st.session_state, 'chat_history'):
                        st.session_state.chat_history = []
                    st.success("Chat history cleared!")
            
            with col2:
                if st.button("📈 Show KPIs", use_container_width=True):
                    kpis = self.sql_agent.get_available_kpis()
                    st.write("**Available KPIs:**")
                    for name, desc in kpis.items():
                        st.write(f"• {name}: {desc}")
            
            if st.button("📊 System Statistics", use_container_width=True):
                self._show_system_stats()
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0; color: #64748b; font-size: 0.8rem;">
                <div style="margin-bottom: 0.5rem;">Powered by</div>
                <div style="font-weight: 600; color: #667eea;">
                    LangChain • OpenAI • FAISS • Streamlit
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_rag_interface(self):
        """Render the RAG document Q&A interface"""
        st.markdown('<div class="section-header">Document Q&A (RAG)</div>', unsafe_allow_html=True)
        
        # Question input - simplified styling
        st.markdown("""
        <div style="font-weight: 600; color: #2d3748; margin-bottom: 1rem; font-size: 1.1rem;">
            Ask a question about enterprise documents:
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we have a selected example question
        if 'selected_example_question' in st.session_state and st.session_state.selected_example_question:
            # Set the rag_question to the selected example
            st.session_state.rag_question = st.session_state.selected_example_question
            # Clear the selected example
            st.session_state.selected_example_question = ""
        
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the warehouse inventory management procedures?",
            key="rag_question"
        )
        
        # Options - simplified
        col1, col2 = st.columns(2)
        with col1:
            use_conversation = st.checkbox("Conversation Mode", value=False, help="Enable conversation history")
        with col2:
            show_sources = st.checkbox("Show Sources", value=True, help="Display source documents")
        
        # Example questions - simplified styling
        st.markdown("""
        <div style="font-weight: 600; color: #0c4a6e; margin: 1.5rem 0 1rem 0; font-size: 1.1rem;">
            Example Questions:
        </div>
        """, unsafe_allow_html=True)
        
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
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_button = st.button("Ask Question", type="primary", key="ask_rag", use_container_width=True)
        
        # Process the question if button is clicked and question exists
        if ask_button and user_question:
            self._process_rag_query(user_question, use_conversation, show_sources)
        elif ask_button and not user_question:
            st.warning("Please enter a question before clicking 'Ask Question'.")
        
        # Display conversation history - simplified
        if hasattr(st.session_state, 'chat_history') and st.session_state.chat_history:
            st.markdown("""
            <div style="font-weight: 600; color: #2d3748; margin: 2rem 0 1rem 0; font-size: 1.1rem;">
                Conversation History
            </div>
            """, unsafe_allow_html=True)
            
            for i, entry in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
                with st.expander(f"Q{i+1}: {entry['question'][:50]}...", expanded=False):
                    st.markdown(f"""
                    <div class="chat-message">
                        <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.5rem;">Question:</div>
                        <div style="margin-bottom: 1rem;">{entry['question']}</div>
                        <div style="font-weight: 600; color: #2d3748; margin-bottom: 0.5rem;">Answer:</div>
                        <div style="margin-bottom: 1rem;">{entry['answer'][:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if entry.get('sources'):
                        st.markdown(f"**Sources:** {len(entry['sources'])} documents")
    
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
        st.markdown("""
        <div style="font-weight: 600; color: #166534; margin-bottom: 1rem; font-size: 1.1rem;">
            Select a KPI to execute:
        </div>
        """, unsafe_allow_html=True)
        
        # Get available KPIs
        kpis = self.sql_agent.get_available_kpis()
        
        # KPI selection with enhanced styling
        selected_kpi = st.selectbox(
            "Choose KPI:",
            options=list(kpis.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()}: {kpis[x]}"
        )
        
        if st.button("Execute KPI", type="primary", use_container_width=True):
            self._execute_kpi(selected_kpi)
        
        # Quick KPI buttons - simplified
        st.markdown("""
        <div style="font-weight: 600; color: #2d3748; margin: 1.5rem 0 1rem 0; font-size: 1.1rem;">
            Quick Actions:
        </div>
        """, unsafe_allow_html=True)
        
        kpi_list = list(kpis.keys())
        cols = st.columns(3)
        
        for i, kpi in enumerate(kpi_list[:6]):  # Show first 6 KPIs
            with cols[i % 3]:
                if st.button(f"{kpi.replace('_', ' ').title()}", key=f"quick_kpi_{i}", use_container_width=True):
                    self._execute_kpi(kpi)
    
    def _render_natural_language_queries(self):
        """Render natural language SQL interface"""
        st.markdown("""
        <div style="font-weight: 600; color: #92400e; margin-bottom: 1rem; font-size: 1.1rem;">
            Ask questions about your business data in natural language:
        </div>
        """, unsafe_allow_html=True)
        
        # Question input
        # Check if we have a selected example question
        if 'selected_sql_question' in st.session_state and st.session_state.selected_sql_question:
            # Set the nl_sql_question to the selected example
            st.session_state.nl_sql_question = st.session_state.selected_sql_question
            # Clear the selected example
            st.session_state.selected_sql_question = ""
        
        user_question = st.text_input(
            "Your business question:",
            placeholder="e.g., What are the total sales by region?",
            key="nl_sql_question"
        )
        
        # Example questions - simplified
        st.markdown("""
        <div style="font-weight: 600; color: #0c4a6e; margin: 1.5rem 0 1rem 0; font-size: 1.1rem;">
            Example Questions:
        </div>
        """, unsafe_allow_html=True)
        
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
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            ask_button = st.button("Ask Question", type="primary", key="ask_nl_sql", use_container_width=True)
        
        # Process the question if button is clicked and question exists
        if ask_button and user_question:
            self._process_nl_sql_query(user_question)
        elif ask_button and not user_question:
            st.warning("Please enter a question before clicking 'Ask Question'.")
    
    def _render_custom_sql(self):
        """Render custom SQL query interface"""
        st.markdown("""
        <div style="font-weight: 600; color: #581c87; margin-bottom: 1rem; font-size: 1.1rem;">
            Execute custom SQL queries directly:
        </div>
        """, unsafe_allow_html=True)
        
        # SQL input
        # Check if we have a selected example query
        if 'selected_custom_sql' in st.session_state and st.session_state.selected_custom_sql:
            # Set the custom_sql to the selected example
            st.session_state.custom_sql = st.session_state.selected_custom_sql
            # Clear the selected example
            st.session_state.selected_custom_sql = ""
        
        sql_query = st.text_area(
            "Enter SQL query:",
            height=100,
            placeholder="SELECT * FROM sales LIMIT 10",
            key="custom_sql"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Execute", type="primary", use_container_width=True):
                self._execute_custom_sql(sql_query)
        
        with col2:
            if st.button("Show Schema", use_container_width=True):
                self._show_database_schema()
        
        # Example queries - simplified
        st.markdown("""
        <div style="font-weight: 600; color: #0c4a6e; margin: 1.5rem 0 1rem 0; font-size: 1.1rem;">
            Example Queries:
        </div>
        """, unsafe_allow_html=True)
        
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
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.5rem; margin-right: 0.5rem;">Q</div>
                    <div style="font-weight: 600; color: #2d3748;">Question:</div>
                </div>
                <div style="margin-bottom: 1.5rem; padding: 0.5rem;">
                    {question}
                </div>
                """, unsafe_allow_html=True)
                
                # Format the answer nicely
                answer_text = result['answer']
                # Convert markdown-style formatting to Streamlit markdown
                answer_text = answer_text.replace('**', '**').replace('*', '*')
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 1.5rem; margin-right: 0.5rem;">A</div>
                    <div style="font-weight: 600; color: #2d3748;">Answer:</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(answer_text)
                
                # Display sources with enhanced styling
                if result.get('sources') and show_sources:
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin: 1.5rem 0 1rem 0;">
                        <div style="font-size: 1.5rem; margin-right: 0.5rem;">S</div>
                        <div style="font-weight: 600; color: #2d3748;">Sources:</div>
                    </div>
                    """, unsafe_allow_html=True)
                    # Deduplicate sources by filename
                    seen_sources = set()
                    unique_sources = []
                    for source in result['sources']:
                        if source['filename'] not in seen_sources:
                            seen_sources.add(source['filename'])
                            unique_sources.append(source)
                    
                    for source in unique_sources:
                        st.markdown(f"""
                        <div class="source-doc">
                            {source['filename']} ({source['document_type']})
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metadata - simplified
                st.markdown("""
                <div style="font-weight: 600; color: #2d3748; margin: 1.5rem 0 1rem 0; font-size: 1.1rem;">
                    Query Metadata:
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", len(result.get('sources', [])))
                with col2:
                    st.metric("Model", result.get('model_used', 'N/A'))
                with col3:
                    st.metric("Response Time", "< 5s")
                
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
        # Create tabs with enhanced styling
        tab1, tab2, tab3 = st.tabs(["Welcome", "Document Q&A", "Business Intelligence"])
        
        with tab1:
            self._render_welcome_page()
        
        with tab2:
            self.render_rag_interface()
        
        with tab3:
            self.render_sql_interface()
        
        # Enhanced footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); margin: 2rem -2rem -2rem -2rem; border-radius: 20px 20px 0 0; border-top: 1px solid #e2e8f0;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #2d3748; margin-bottom: 0.5rem;">
                Enterprise Knowledge Assistant
            </div>
            <div style="font-size: 1rem; color: #64748b; margin-bottom: 1rem;">
                AI-Powered Document Intelligence & Business Analytics
            </div>
            <div style="font-size: 0.9rem; color: #94a3b8; font-weight: 500;">
                Powered by LangChain • OpenAI • FAISS • Streamlit • FastAPI
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_welcome_page(self):
        """Render the welcome/introduction page"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -2rem -2rem 2rem -2rem; border-radius: 0 0 20px 20px;">
            <div style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                Enterprise Knowledge Assistant
            </div>
            <div style="font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-bottom: 1rem; font-weight: 300;">
                Your AI-Powered Document Intelligence & Business Analytics Platform
            </div>
            <div style="width: 60px; height: 4px; background: linear-gradient(90deg, #ffd89b, #19547b); margin: 0 auto; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # System Overview
        st.markdown("""
        <div style="font-size: 2rem; font-weight: 700; color: #2d3748; margin: 3rem 0 2rem 0; text-align: center; position: relative;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">What This System Does</span>
            <div style="width: 80px; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto 0 auto; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #0ea5e9; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #0c4a6e; margin-bottom: 1rem; font-size: 1.3rem;">Document Intelligence</div>
                <div style="color: #0c4a6e; font-size: 1rem; line-height: 1.6;">
                    • Ask questions about enterprise documents<br>
                    • Get AI-powered answers with sources<br>
                    • Search through SOPs, reports, and logs<br>
                    • Maintain conversation history
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #10b981; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #166534; margin-bottom: 1rem; font-size: 1.3rem;">Smart Search</div>
                <div style="color: #166534; font-size: 1rem; line-height: 1.6;">
                    • Semantic document search<br>
                    • Context-aware answers<br>
                    • Source document tracking<br>
                    • Relevance scoring
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #f59e0b; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #92400e; margin-bottom: 1rem; font-size: 1.3rem;">Business Analytics</div>
                <div style="color: #92400e; font-size: 1rem; line-height: 1.6;">
                    • KPI dashboard and metrics<br>
                    • Natural language SQL queries<br>
                    • Custom SQL execution<br>
                    • Data visualization
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); padding: 2rem; border-radius: 16px; border-left: 6px solid #a855f7; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; color: #581c87; margin-bottom: 1rem; font-size: 1.3rem;">AI Integration</div>
                <div style="color: #581c87; font-size: 1rem; line-height: 1.6;">
                    • OpenAI GPT-4 integration<br>
                    • LangChain framework<br>
                    • FAISS vector search<br>
                    • Intelligent responses
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # How to Use
        st.markdown("""
        <div style="font-size: 2rem; font-weight: 700; color: #2d3748; margin: 3rem 0 2rem 0; text-align: center; position: relative;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">How to Use</span>
            <div style="width: 80px; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto 0 auto; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; transition: transform 0.2s;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-right: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">1</div>
                <div style="font-weight: 700; color: #2d3748; font-size: 1.3rem;">Document Q&A</div>
            </div>
            <div style="color: #64748b; margin-left: 3rem; font-size: 1rem; line-height: 1.6;">
                Go to the "Document Q&A" tab to ask questions about enterprise documents. Try example questions or ask your own!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1.5rem; transition: transform 0.2s;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 2rem; margin-right: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">2</div>
                <div style="font-weight: 700; color: #2d3748; font-size: 1.3rem;">Business Intelligence</div>
            </div>
            <div style="color: #64748b; margin-left: 3rem; font-size: 1rem; line-height: 1.6;">
                Visit the "Business Intelligence" tab to explore KPIs, run natural language queries, or execute custom SQL.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("""
        <div style="font-size: 2rem; font-weight: 700; color: #2d3748; margin: 3rem 0 2rem 0; text-align: center; position: relative;">
            <span style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">System Status</span>
            <div style="width: 80px; height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto 0 auto; border-radius: 2px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        rag_status = st.session_state.system_status.get('rag_status', {})
        sql_status = st.session_state.system_status.get('sql_status', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "#10b981" if rag_status.get('vector_store_loaded') else "#f59e0b"
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">DB</div>
                <div style="font-weight: 700; color: {status_color}; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    {'Ready' if rag_status.get('vector_store_loaded') else 'Loading'}
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">Document Search</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_color = "#10b981" if sql_status.get('database_connected') else "#f59e0b"
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">DB</div>
                <div style="font-weight: 700; color: {status_color}; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    {'Connected' if sql_status.get('database_connected') else 'Disconnected'}
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">Database</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status_color = "#10b981" if rag_status.get('llm_available') else "#f59e0b"
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">AI</div>
                <div style="font-weight: 700; color: {status_color}; margin-bottom: 0.5rem; font-size: 1.1rem;">
                    {'Available' if rag_status.get('llm_available') else 'Limited'}
                </div>
                <div style="font-size: 0.9rem; color: #64748b;">AI Model</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status_color = "#10b981" if sql_status.get('sql_agent_available') else "#f59e0b"
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); border-radius: 16px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700;">SQL</div>
                <div style="font-size: 0.9rem; color: #64748b;">SQL Agent</div>
            </div>
            """, unsafe_allow_html=True)
    
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
        # Initialize session state
        self._initialize_session_state()
        
        # Load system components
        self._load_system_components()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main interface
        self.render_main_interface()


def main():
    """Main function to run the dashboard"""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 