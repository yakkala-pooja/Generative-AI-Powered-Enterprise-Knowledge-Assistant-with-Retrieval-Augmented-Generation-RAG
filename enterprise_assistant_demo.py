#!/usr/bin/env python3
"""
Enterprise Knowledge Assistant - Comprehensive Demo

This script demonstrates the complete integration of:
- LangChain RAG pipeline with GPT-4
- SQL agent for database queries and KPIs
- Streamlit dashboard interface
- FastAPI backend API

Usage:
    python enterprise_assistant_demo.py [--mode interactive|api|streamlit|all]
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_rag_pipeline import LangChainRAGPipeline
    from sql_agent import EnterpriseSQLAgent
    from contoso_abf_loader import ContosoABFLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the same directory")
    sys.exit(1)

class EnterpriseAssistantDemo:
    """
    Comprehensive demo class for the Enterprise Knowledge Assistant
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the demo system
        
        Args:
            openai_api_key: OpenAI API key for LLM features
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.rag_pipeline = None
        self.sql_agent = None
        self.demo_results = {}
        
        print("Enterprise Knowledge Assistant - Comprehensive Demo")
        print("=" * 60)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        print("\n1. Initializing System Components...")
        
        try:
            # Initialize RAG pipeline
            print("   Initializing RAG pipeline...")
            self.rag_pipeline = LangChainRAGPipeline(
                openai_api_key=self.openai_api_key,
                embedding_model="sentence-transformers",
                llm_model="gpt-4"
            )
            
            # Initialize SQL agent
            print("   Initializing SQL agent...")
            self.sql_agent = EnterpriseSQLAgent(
                openai_api_key=self.openai_api_key,
                use_sqlite=True
            )
            
            # Check system status
            rag_status = self.rag_pipeline.get_system_status()
            sql_status = self.sql_agent.get_system_status()
            
            print(f"   RAG Pipeline: {'Ready' if rag_status['vector_store_loaded'] else 'Limited'}")
            print(f"   SQL Agent: {'Ready' if sql_status['database_connected'] else 'Limited'}")
            print(f"   LLM: {'Available' if rag_status['llm_available'] else 'Not Available'}")
            
            self.demo_results['initialization'] = {
                'rag_status': rag_status,
                'sql_status': sql_status,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"   Error initializing components: {e}")
            print("   Note: Some features may be limited without proper API keys")
    
    def demo_rag_pipeline(self):
        """Demonstrate RAG pipeline capabilities"""
        print("\n2. RAG Pipeline Demonstration")
        print("-" * 40)
        
        if not self.rag_pipeline:
            print("   RAG pipeline not available")
            return
        
        # Demo queries
        demo_queries = [
            {
                "question": "What are the standard procedures for warehouse inventory management?",
                "category": "SOPs"
            },
            {
                "question": "How should we handle supplier quality issues?",
                "category": "Quality Management"
            },
            {
                "question": "What are the temperature requirements for cold chain management?",
                "category": "Cold Chain"
            },
            {
                "question": "How do we resolve customer delivery delays?",
                "category": "Customer Service"
            }
        ]
        
        rag_results = []
        
        for i, query_info in enumerate(demo_queries, 1):
            print(f"\n   Query {i}: {query_info['question']}")
            print(f"   Category: {query_info['category']}")
            
            try:
                result = self.rag_pipeline.query(
                    question=query_info['question'],
                    use_conversation=False
                )
                
                print(f"   Answer: {result['answer'][:150]}...")
                print(f"   Sources: {result['source_count']} documents")
                print(f"   Model: {result['model_used']}")
                
                rag_results.append({
                    'query': query_info,
                    'result': result,
                    'success': True
                })
                
            except Exception as e:
                print(f"   Error: {e}")
                rag_results.append({
                    'query': query_info,
                    'error': str(e),
                    'success': False
                })
        
        self.demo_results['rag_demo'] = rag_results
        
        # Test conversational capability
        print(f"\n   Testing Conversational RAG:")
        try:
            # First question
            result1 = self.rag_pipeline.query(
                "What are warehouse safety procedures?", 
                use_conversation=True
            )
            print(f"   Q1: What are warehouse safety procedures?")
            print(f"   A1: {result1['answer'][:100]}...")
            
            # Follow-up question
            result2 = self.rag_pipeline.query(
                "What about inventory accuracy requirements?", 
                use_conversation=True
            )
            print(f"   Q2: What about inventory accuracy requirements?")
            print(f"   A2: {result2['answer'][:100]}...")
            
            print(f"   Conversational memory working")
            
        except Exception as e:
            print(f"   Conversational error: {e}")
    
    def demo_sql_agent(self):
        """Demonstrate SQL agent capabilities"""
        print("\n3. SQL Agent Demonstration")
        print("-" * 40)
        
        if not self.sql_agent:
            print("   SQL agent not available")
            return
        
        # Show available KPIs
        print("   Available KPIs:")
        try:
            kpis = self.sql_agent.get_available_kpis()
            for name, description in list(kpis.items())[:3]:  # Show first 3
                print(f"     - {name}: {description}")
            print(f"     ... and {len(kpis) - 3} more")
        except Exception as e:
            print(f"   Error getting KPIs: {e}")
            return
        
        # Execute sample KPIs
        sample_kpis = ["sales_by_region", "revenue_by_product_category", "top_products"]
        kpi_results = []
        
        for kpi_name in sample_kpis:
            print(f"\n   Executing KPI: {kpi_name}")
            try:
                result = self.sql_agent.execute_kpi_query(kpi_name)
                
                if "error" in result:
                    print(f"   Error: {result['error']}")
                    kpi_results.append({'kpi': kpi_name, 'success': False, 'error': result['error']})
                else:
                    print(f"   Success: {result['row_count']} rows returned")
                    print(f"   Columns: {', '.join(result['columns'][:3])}{'...' if len(result['columns']) > 3 else ''}")
                    
                    # Show sample data
                    if result['data']:
                        sample_row = result['data'][0]
                        sample_str = ", ".join([f"{k}: {v}" for k, v in list(sample_row.items())[:2]])
                        print(f"   Sample: {sample_str}...")
                    
                    kpi_results.append({'kpi': kpi_name, 'success': True, 'result': result})
                    
            except Exception as e:
                print(f"   Error executing {kpi_name}: {e}")
                kpi_results.append({'kpi': kpi_name, 'success': False, 'error': str(e)})
        
        # Test natural language queries
        print(f"\n   Testing Natural Language Queries:")
        nl_queries = [
            "What are the total sales by region?",
            "Which product categories generate the most revenue?",
            "Show me the top 5 products by sales"
        ]
        
        nl_results = []
        for query in nl_queries:
            print(f"   Query: {query}")
            try:
                result = self.sql_agent.query_with_agent(query)
                print(f"   Answer: {result['answer'][:100]}...")
                print(f"   Agent Used: {'Yes' if result['agent_used'] else 'No'}")
                nl_results.append({'query': query, 'success': True, 'result': result})
                
            except Exception as e:
                print(f"   Error: {e}")
                nl_results.append({'query': query, 'success': False, 'error': str(e)})
        
        self.demo_results['sql_demo'] = {
            'kpi_results': kpi_results,
            'nl_results': nl_results
        }
    
    def demo_integration_scenarios(self):
        """Demonstrate integrated scenarios combining RAG and SQL"""
        print("\n4. Integration Scenarios")
        print("-" * 40)
        
        scenarios = [
            {
                "name": "Supply Chain Analysis",
                "rag_query": "What are the supplier qualification procedures?",
                "sql_query": "revenue_by_product_category"
            },
            {
                "name": "Quality Management",
                "rag_query": "How do we handle quality defects?",
                "sql_query": "top_products"
            },
            {
                "name": "Performance Review",
                "rag_query": "What are the key supply chain KPIs?",
                "sql_query": "monthly_sales_trend"
            }
        ]
        
        integration_results = []
        
        for scenario in scenarios:
            print(f"\n   Scenario: {scenario['name']}")
            
            scenario_result = {'name': scenario['name']}
            
            # RAG component
            if self.rag_pipeline:
                try:
                    rag_result = self.rag_pipeline.query(scenario['rag_query'])
                    print(f"   Document insight: {rag_result['answer'][:80]}...")
                    scenario_result['rag_success'] = True
                    scenario_result['rag_sources'] = rag_result['source_count']
                except Exception as e:
                    print(f"   RAG error: {e}")
                    scenario_result['rag_success'] = False
            
            # SQL component
            if self.sql_agent:
                try:
                    sql_result = self.sql_agent.execute_kpi_query(scenario['sql_query'])
                    if "error" not in sql_result:
                        print(f"   Data insight: {sql_result['row_count']} records analyzed")
                        scenario_result['sql_success'] = True
                        scenario_result['sql_rows'] = sql_result['row_count']
                    else:
                        print(f"   SQL error: {sql_result['error']}")
                        scenario_result['sql_success'] = False
                except Exception as e:
                    print(f"   SQL error: {e}")
                    scenario_result['sql_success'] = False
            
            integration_results.append(scenario_result)
        
        self.demo_results['integration_demo'] = integration_results
    
    def demo_api_endpoints(self):
        """Demonstrate API functionality (without starting server)"""
        print("\n5. API Endpoints Overview")
        print("-" * 40)
        
        # Show what the API would provide
        api_endpoints = {
            "RAG Endpoints": [
                "POST /api/v1/rag/query - Query documents with natural language",
                "GET /api/v1/rag/conversation-history - Get conversation history",
                "DELETE /api/v1/rag/conversation-history - Clear conversation history"
            ],
            "SQL Endpoints": [
                "POST /api/v1/sql/query - Natural language data queries",
                "GET /api/v1/sql/kpis - List available KPIs",
                "POST /api/v1/sql/kpi - Execute specific KPI",
                "POST /api/v1/sql/custom - Execute custom SQL",
                "GET /api/v1/sql/schema - Get database schema"
            ],
            "System Endpoints": [
                "GET /health - Health check",
                "GET /api/v1/status - System status",
                "GET /api/v1/analytics/usage - Usage analytics"
            ]
        }
        
        for category, endpoints in api_endpoints.items():
            print(f"\n   {category}:")
            for endpoint in endpoints:
                print(f"     - {endpoint}")
        
        print(f"\n   API Documentation: http://localhost:8000/docs")
        print(f"   ReDoc: http://localhost:8000/redoc")
    
    def run_interactive_mode(self):
        """Run interactive demo mode"""
        print("\n6. Interactive Mode")
        print("-" * 40)
        
        while True:
            print(f"\nOptions:")
            print("1. Ask a document question (RAG)")
            print("2. Execute a KPI query (SQL)")
            print("3. Ask a data question (SQL)")
            print("4. Show system status")
            print("5. Exit")
            
            try:
                choice = input("\nSelect an option (1-5): ").strip()
                
                if choice == "1":
                    self._interactive_rag()
                elif choice == "2":
                    self._interactive_kpi()
                elif choice == "3":
                    self._interactive_sql()
                elif choice == "4":
                    self._show_system_status()
                elif choice == "5":
                    print("Goodbye! 👋")
                    break
                else:
                    print("Invalid option. Please try again.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _interactive_rag(self):
        """Interactive RAG demo"""
        if not self.rag_pipeline:
            print("RAG pipeline not available")
            return
        
        question = input("Ask a question about enterprise documents: ").strip()
        if not question:
            return
        
        try:
            result = self.rag_pipeline.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {result['source_count']} documents")
            
            if result.get('sources'):
                print(f"\nSource Documents:")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. {source['filename']} ({source['document_type']})")
                    
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_kpi(self):
        """Interactive KPI demo"""
        if not self.sql_agent:
            print("SQL agent not available")
            return
        
        try:
            kpis = self.sql_agent.get_available_kpis()
            print("\nAvailable KPIs:")
            kpi_list = list(kpis.keys())
            
            for i, (name, desc) in enumerate(kpis.items(), 1):
                print(f"  {i}. {name}: {desc}")
            
            choice = input(f"\nSelect KPI (1-{len(kpis)}): ").strip()
            
            try:
                index = int(choice) - 1
                if 0 <= index < len(kpi_list):
                    kpi_name = kpi_list[index]
                    result = self.sql_agent.execute_kpi_query(kpi_name)
                    
                    if "error" in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\nKPI: {result['kpi_name']}")
                        print(f"Description: {result['description']}")
                        print(f"Rows: {result['row_count']}")
                        
                        if result['data']:
                            print(f"\nSample Data:")
                            for i, row in enumerate(result['data'][:3], 1):
                                row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:3]])
                                print(f"  {i}. {row_str}...")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Please enter a valid number")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_sql(self):
        """Interactive SQL demo"""
        if not self.sql_agent:
            print("SQL agent not available")
            return
        
        question = input("Ask a question about your business data: ").strip()
        if not question:
            return
        
        try:
            result = self.sql_agent.query_with_agent(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Agent Used: {'Yes' if result['agent_used'] else 'No'}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def _show_system_status(self):
        """Show current system status"""
        print(f"\nSystem Status:")
        
        if self.rag_pipeline:
            rag_status = self.rag_pipeline.get_system_status()
            print(f"RAG Pipeline:")
            print(f"  - Vector Store: {'Ready' if rag_status['vector_store_loaded'] else 'Limited'}")
            print(f"  - LLM: {'Available' if rag_status['llm_available'] else 'Not Available'}")
            print(f"  - Model: {rag_status['llm_model']}")
            
            if rag_status.get('vector_store_stats'):
                vs_stats = rag_status['vector_store_stats']
                print(f"  - Documents: {vs_stats['total_vectors']:,}")
        
        if self.sql_agent:
            sql_status = self.sql_agent.get_system_status()
            print(f"SQL Agent:")
            print(f"  - Database: {'Connected' if sql_status['database_connected'] else 'Not Connected'}")
            print(f"  - Agent: {'Available' if sql_status['sql_agent_available'] else 'Not Available'}")
            print(f"  - KPIs: {sql_status['available_kpis']}")
    
    def generate_demo_report(self):
        """Generate a comprehensive demo report"""
        print("\n7. Demo Report Generation")
        print("-" * 40)
        
        report = {
            "demo_summary": {
                "timestamp": datetime.now().isoformat(),
                "components_tested": ["RAG Pipeline", "SQL Agent", "Integration"],
                "total_queries": 0,
                "successful_queries": 0
            },
            "detailed_results": self.demo_results
        }
        
        # Calculate summary statistics
        if 'rag_demo' in self.demo_results:
            rag_queries = len(self.demo_results['rag_demo'])
            rag_success = sum(1 for r in self.demo_results['rag_demo'] if r['success'])
            report["demo_summary"]["total_queries"] += rag_queries
            report["demo_summary"]["successful_queries"] += rag_success
        
        if 'sql_demo' in self.demo_results:
            if 'kpi_results' in self.demo_results['sql_demo']:
                kpi_queries = len(self.demo_results['sql_demo']['kpi_results'])
                kpi_success = sum(1 for r in self.demo_results['sql_demo']['kpi_results'] if r['success'])
                report["demo_summary"]["total_queries"] += kpi_queries
                report["demo_summary"]["successful_queries"] += kpi_success
        
        # Save report
        report_file = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n7. Demo Report Generation")
        print("-" * 40)
        print(f"   Demo report saved to: {report_file}")
        
        # Generate summary
        total_queries = 0
        successful_queries = 0
        
        if 'rag_demo' in self.demo_results:
            total_queries += len(self.demo_results['rag_demo'])
            successful_queries += sum(1 for r in self.demo_results['rag_demo'] if r.get('success', False))
        
        if 'sql_demo' in self.demo_results:
            sql_demo = self.demo_results['sql_demo']
            if 'kpi_results' in sql_demo:
                total_queries += len(sql_demo['kpi_results'])
                successful_queries += sum(1 for r in sql_demo['kpi_results'] if r.get('success', False))
            if 'nl_results' in sql_demo:
                total_queries += len(sql_demo['nl_results'])
                successful_queries += sum(1 for r in sql_demo['nl_results'] if r.get('success', False))
        
        summary = {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0
        }
        
        print(f"   Total Queries: {summary['total_queries']}")
        print(f"   Successful: {summary['successful_queries']}")
        print(f"   Success Rate: {(summary['successful_queries']/summary['total_queries']*100):.1f}%" if summary['total_queries'] > 0 else "   Success Rate: N/A")
        
        print(f"\nDemo Complete!")
        print("Next steps:")
        print("  - Run 'python enterprise_assistant_demo.py --mode interactive' for hands-on testing")
        print("  - Run 'python enterprise_assistant_demo.py --mode streamlit' for web interface")
        print("  - Run 'python enterprise_assistant_demo.py --mode api' for API server")


def run_streamlit_dashboard():
    """Start Streamlit dashboard"""
    print("Starting Streamlit dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_dashboard.py", "--server.port", "8501"
    ])


def run_fastapi_server():
    """Start FastAPI server"""
    print("Starting FastAPI server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enterprise Knowledge Assistant Demo")
    parser.add_argument("--mode", choices=["demo", "interactive", "streamlit", "api", "all"], 
                       default="demo", help="Demo mode")
    parser.add_argument("--openai-key", help="OpenAI API key")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo = EnterpriseAssistantDemo(openai_api_key=args.openai_key)
        demo.run_comprehensive_demo()
    
    elif args.mode == "interactive":
        demo = EnterpriseAssistantDemo(openai_api_key=args.openai_key)
        demo.run_interactive_mode()
    
    elif args.mode == "streamlit":
        run_streamlit_dashboard()
    
    elif args.mode == "api":
        run_fastapi_server()
    
    elif args.mode == "all":
        print("Starting all services...")
        # Start Streamlit in background
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py", "--server.port", "8501"
        ])
        
        # Start FastAPI in background
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000"
        ])
        
        print("All services started!")
        print("Streamlit Dashboard: http://localhost:8501")
        print("FastAPI Server: http://localhost:8000")
        print("Press Ctrl+C to stop all services")
        
        try:
            streamlit_process.wait()
            api_process.wait()
        except KeyboardInterrupt:
            print("\nStopping services...")
            streamlit_process.terminate()
            api_process.terminate()
            print("Services stopped")


if __name__ == "__main__":
    main() 