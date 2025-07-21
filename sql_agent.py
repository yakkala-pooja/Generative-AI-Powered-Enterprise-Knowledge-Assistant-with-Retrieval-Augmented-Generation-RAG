#!/usr/bin/env python3
"""
SQL Agent for Enterprise Database Operations

This module implements a LangChain SQL agent for querying Contoso sales database
and extracting KPIs, metrics, and business intelligence insights.
"""

import os
import json
import sqlite3
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Database imports
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker

# LangChain imports
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.schema import AgentAction, AgentFinish

# Local imports
from contoso_abf_loader import ContosoABFLoader

class EnterpriseSQLAgent:
    """
    Enterprise SQL Agent for database operations and KPI extraction
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
        use_sqlite: bool = True
    ):
        """
        Initialize the SQL agent
        
        Args:
            database_url: Database connection URL
            openai_api_key: OpenAI API key for LLM
            llm_model: LLM model to use
            use_sqlite: Whether to use SQLite for demo (True) or external DB (False)
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_model = llm_model
        self.use_sqlite = use_sqlite
        
        # Database setup
        if use_sqlite:
            self.database_url = database_url or "sqlite:///contoso_enterprise.db"
        else:
            self.database_url = database_url
        
        self.engine = None
        self.db_connection = None
        self.sql_agent = None
        
        # Initialize LLM
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name=llm_model,
                temperature=0.1
            )
        else:
            print("Warning: No OpenAI API key provided. SQL agent features will be limited.")
            self.llm = None
        
        # Setup database and agent
        self._setup_database()
        self._setup_sql_agent()
        
        # KPI definitions
        self.kpi_definitions = self._define_kpis()
    
    def _setup_database(self):
        """Setup database connection and populate with Contoso data"""
        print("Setting up database connection...")
        
        try:
            self.engine = create_engine(self.database_url, echo=False)
            
            if self.use_sqlite:
                # Check if database exists and has data
                inspector = inspect(self.engine)
                existing_tables = inspector.get_table_names()
                
                if not existing_tables or 'sales' not in existing_tables:
                    print("Populating database with Contoso sample data...")
                    self._populate_sample_data()
                else:
                    print("Using existing database data")
            
            # Create SQL Database wrapper for LangChain
            self.db_connection = SQLDatabase.from_uri(self.database_url)
            print("Database connection established successfully")
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            self.engine = None
            self.db_connection = None
    
    def _populate_sample_data(self):
        """Populate database with sample Contoso data"""
        try:
            # Generate sample data using ABF loader
            abf_loader = ContosoABFLoader("Data/Contoso_Retail.abf")
            sample_data = abf_loader.method_4_generate_sample_data()
            
            if sample_data:
                # Store data in database
                for table_name, df in sample_data.items():
                    df.to_sql(
                        table_name.lower(),
                        self.engine,
                        if_exists='replace',
                        index=False
                    )
                    print(f"Created table '{table_name.lower()}' with {len(df)} rows")
                
                # Create additional views and indexes for better performance
                self._create_database_views()
                print("Sample data populated successfully")
            else:
                print("Failed to generate sample data")
                
        except Exception as e:
            print(f"Error populating sample data: {e}")
    
    def _create_database_views(self):
        """Create useful database views for KPI queries"""
        try:
            with self.engine.connect() as conn:
                # Sales summary view
                sales_summary_query = """
                CREATE VIEW IF NOT EXISTS sales_summary AS
                SELECT 
                    s.DateKey,
                    d.Year,
                    d.Quarter,
                    d.Month,
                    d.MonthName,
                    p.Category,
                    p.Subcategory,
                    c.City,
                    c.StateProvince,
                    COUNT(s.SalesKey) as TransactionCount,
                    SUM(s.Quantity) as TotalQuantity,
                    SUM(s.SalesAmount) as TotalSalesAmount,
                    SUM(s.TotalCost) as TotalCost,
                    SUM(s.Profit) as TotalProfit,
                    AVG(s.UnitPrice) as AvgUnitPrice
                FROM sales s
                JOIN date d ON s.DateKey = d.DateKey
                JOIN product p ON s.ProductKey = p.ProductKey
                JOIN customer c ON s.CustomerKey = c.CustomerKey
                GROUP BY s.DateKey, d.Year, d.Quarter, d.Month, d.MonthName, 
                         p.Category, p.Subcategory, c.City, c.StateProvince
                """
                
                # Monthly performance view
                monthly_performance_query = """
                CREATE VIEW IF NOT EXISTS monthly_performance AS
                SELECT 
                    d.Year,
                    d.Month,
                    d.MonthName,
                    COUNT(DISTINCT s.CustomerKey) as UniqueCustomers,
                    COUNT(s.SalesKey) as TotalTransactions,
                    SUM(s.SalesAmount) as MonthlyRevenue,
                    SUM(s.Profit) as MonthlyProfit,
                    AVG(s.SalesAmount) as AvgTransactionValue,
                    SUM(s.SalesAmount) / COUNT(DISTINCT s.CustomerKey) as RevenuePerCustomer
                FROM sales s
                JOIN date d ON s.DateKey = d.DateKey
                GROUP BY d.Year, d.Month, d.MonthName
                ORDER BY d.Year, d.Month
                """
                
                # Product performance view
                product_performance_query = """
                CREATE VIEW IF NOT EXISTS product_performance AS
                SELECT 
                    p.ProductKey,
                    p.ProductName,
                    p.Category,
                    p.Subcategory,
                    COUNT(s.SalesKey) as TransactionCount,
                    SUM(s.Quantity) as TotalQuantitySold,
                    SUM(s.SalesAmount) as TotalRevenue,
                    SUM(s.Profit) as TotalProfit,
                    AVG(s.UnitPrice) as AvgSellingPrice,
                    (SUM(s.Profit) / SUM(s.SalesAmount)) * 100 as ProfitMarginPercent
                FROM sales s
                JOIN product p ON s.ProductKey = p.ProductKey
                GROUP BY p.ProductKey, p.ProductName, p.Category, p.Subcategory
                """
                
                # Customer segmentation view
                customer_segmentation_query = """
                CREATE VIEW IF NOT EXISTS customer_segmentation AS
                SELECT 
                    c.CustomerKey,
                    c.FirstName || ' ' || c.LastName as CustomerName,
                    c.City,
                    c.StateProvince,
                    c.YearlyIncome,
                    c.Education,
                    c.Occupation,
                    COUNT(s.SalesKey) as TotalTransactions,
                    SUM(s.SalesAmount) as TotalSpent,
                    AVG(s.SalesAmount) as AvgTransactionValue,
                    MAX(d.Date) as LastPurchaseDate,
                    CASE 
                        WHEN SUM(s.SalesAmount) > 5000 THEN 'High Value'
                        WHEN SUM(s.SalesAmount) > 2000 THEN 'Medium Value'
                        ELSE 'Low Value'
                    END as CustomerSegment
                FROM sales s
                JOIN customer c ON s.CustomerKey = c.CustomerKey
                JOIN date d ON s.DateKey = d.DateKey
                GROUP BY c.CustomerKey, c.FirstName, c.LastName, c.City, 
                         c.StateProvince, c.YearlyIncome, c.Education, c.Occupation
                """
                
                # Execute view creation queries
                for query in [sales_summary_query, monthly_performance_query, 
                            product_performance_query, customer_segmentation_query]:
                    conn.execute(text(query))
                    conn.commit()
                
                print("Database views created successfully")
                
        except Exception as e:
            print(f"Error creating database views: {e}")
    
    def _setup_sql_agent(self):
        """Setup LangChain SQL agent"""
        if not self.llm or not self.db_connection:
            print("Cannot setup SQL agent: Missing LLM or database connection")
            return
        
        try:
            print("Setting up SQL agent...")
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(
                db=self.db_connection,
                llm=self.llm
            )
            
            # Create SQL agent
            self.sql_agent = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )
            
            print("SQL agent setup complete")
            
        except Exception as e:
            print(f"Error setting up SQL agent: {e}")
            self.sql_agent = None
    
    def _define_kpis(self) -> Dict[str, Dict[str, str]]:
        """Define standard KPIs and their SQL queries"""
        return {
            "sales_by_region": {
                "description": "Total sales amount by region/state",
                "query": """
                SELECT 
                    c.StateProvince as Region,
                    SUM(s.SalesAmount) as TotalSales,
                    COUNT(s.SalesKey) as TransactionCount,
                    COUNT(DISTINCT s.CustomerKey) as UniqueCustomers
                FROM sales s
                JOIN customer c ON s.CustomerKey = c.CustomerKey
                GROUP BY c.StateProvince
                ORDER BY TotalSales DESC
                """
            },
            
            "revenue_by_product_category": {
                "description": "Revenue breakdown by product category",
                "query": """
                SELECT 
                    p.Category,
                    SUM(s.SalesAmount) as TotalRevenue,
                    SUM(s.Profit) as TotalProfit,
                    COUNT(s.SalesKey) as TransactionCount,
                    AVG(s.UnitPrice) as AvgUnitPrice
                FROM sales s
                JOIN product p ON s.ProductKey = p.ProductKey
                GROUP BY p.Category
                ORDER BY TotalRevenue DESC
                """
            },
            
            "monthly_sales_trend": {
                "description": "Monthly sales trend analysis",
                "query": """
                SELECT 
                    d.Year,
                    d.Month,
                    d.MonthName,
                    SUM(s.SalesAmount) as MonthlySales,
                    SUM(s.Profit) as MonthlyProfit,
                    COUNT(s.SalesKey) as TransactionCount
                FROM sales s
                JOIN date d ON s.DateKey = d.DateKey
                GROUP BY d.Year, d.Month, d.MonthName
                ORDER BY d.Year, d.Month
                """
            },
            
            "top_products": {
                "description": "Top 10 best-selling products by revenue",
                "query": """
                SELECT 
                    p.ProductName,
                    p.Category,
                    p.Subcategory,
                    SUM(s.SalesAmount) as TotalRevenue,
                    SUM(s.Quantity) as TotalQuantitySold,
                    COUNT(s.SalesKey) as TransactionCount
                FROM sales s
                JOIN product p ON s.ProductKey = p.ProductKey
                GROUP BY p.ProductKey, p.ProductName, p.Category, p.Subcategory
                ORDER BY TotalRevenue DESC
                LIMIT 10
                """
            },
            
            "customer_lifetime_value": {
                "description": "Customer lifetime value analysis",
                "query": """
                SELECT 
                    c.CustomerKey,
                    c.FirstName || ' ' || c.LastName as CustomerName,
                    c.City,
                    c.YearlyIncome,
                    SUM(s.SalesAmount) as LifetimeValue,
                    COUNT(s.SalesKey) as TotalTransactions,
                    AVG(s.SalesAmount) as AvgTransactionValue,
                    MAX(d.Date) as LastPurchaseDate
                FROM sales s
                JOIN customer c ON s.CustomerKey = c.CustomerKey
                JOIN date d ON s.DateKey = d.DateKey
                GROUP BY c.CustomerKey, c.FirstName, c.LastName, c.City, c.YearlyIncome
                ORDER BY LifetimeValue DESC
                LIMIT 20
                """
            },
            
            "supply_chain_performance": {
                "description": "Supply chain and inventory performance metrics",
                "query": """
                SELECT 
                    p.Category,
                    COUNT(DISTINCT p.ProductKey) as ProductCount,
                    SUM(s.Quantity) as TotalUnitsSold,
                    AVG(s.UnitPrice) as AvgSellingPrice,
                    AVG(p.StandardCost) as AvgCost,
                    (AVG(s.UnitPrice) - AVG(p.StandardCost)) / AVG(s.UnitPrice) * 100 as AvgMarginPercent
                FROM sales s
                JOIN product p ON s.ProductKey = p.ProductKey
                GROUP BY p.Category
                ORDER BY TotalUnitsSold DESC
                """
            }
        }
    
    def execute_kpi_query(self, kpi_name: str) -> Dict[str, Any]:
        """
        Execute a predefined KPI query
        
        Args:
            kpi_name: Name of the KPI to execute
            
        Returns:
            Query results and metadata
        """
        if kpi_name not in self.kpi_definitions:
            return {
                "error": f"Unknown KPI: {kpi_name}",
                "available_kpis": list(self.kpi_definitions.keys())
            }
        
        if not self.engine:
            return {"error": "Database connection not available"}
        
        try:
            kpi_info = self.kpi_definitions[kpi_name]
            
            print(f"Executing KPI query: {kpi_name}")
            
            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(kpi_info["query"]))
                columns = result.keys()
                rows = result.fetchall()
            
            # Convert to list of dictionaries
            data = [dict(zip(columns, row)) for row in rows]
            
            return {
                "kpi_name": kpi_name,
                "description": kpi_info["description"],
                "data": data,
                "row_count": len(data),
                "columns": list(columns),
                "execution_time": datetime.now().isoformat(),
                "query": kpi_info["query"]
            }
            
        except Exception as e:
            return {
                "error": f"Error executing KPI query: {e}",
                "kpi_name": kpi_name
            }
    
    def query_with_agent(self, question: str) -> Dict[str, Any]:
        """
        Query the database using the LangChain SQL agent
        
        Args:
            question: Natural language question about the data
            
        Returns:
            Agent response and metadata
        """
        if not self.sql_agent:
            return self._fallback_direct_query(question)
        
        try:
            print(f"Processing SQL agent query: {question}")
            
            # Execute query with agent
            result = self.sql_agent.invoke({"input": question})
            
            return {
                "question": question,
                "answer": result.get("output", ""),
                "agent_used": True,
                "timestamp": datetime.now().isoformat(),
                "model": self.llm_model
            }
            
        except Exception as e:
            print(f"Error with SQL agent: {e}")
            return self._fallback_direct_query(question)
    
    def _fallback_direct_query(self, question: str) -> Dict[str, Any]:
        """
        Fallback method for direct SQL queries when agent is not available
        """
        # Map common questions to KPIs
        question_lower = question.lower()
        
        kpi_mapping = {
            "sales by region": "sales_by_region",
            "revenue by category": "revenue_by_product_category",
            "monthly sales": "monthly_sales_trend",
            "top products": "top_products",
            "customer value": "customer_lifetime_value",
            "supply chain": "supply_chain_performance"
        }
        
        for phrase, kpi in kpi_mapping.items():
            if phrase in question_lower:
                result = self.execute_kpi_query(kpi)
                if "error" not in result:
                    return {
                        "question": question,
                        "answer": f"Based on your question, I executed the '{kpi}' KPI query. Results show {result['row_count']} records.",
                        "kpi_result": result,
                        "agent_used": False,
                        "timestamp": datetime.now().isoformat()
                    }
        
        return {
            "question": question,
            "answer": "I couldn't process your question. Please try one of the predefined KPI queries.",
            "available_kpis": list(self.kpi_definitions.keys()),
            "agent_used": False,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_available_kpis(self) -> Dict[str, str]:
        """
        Get list of available KPIs and their descriptions
        
        Returns:
            Dictionary of KPI names and descriptions
        """
        return {
            name: info["description"] 
            for name, info in self.kpi_definitions.items()
        }
    
    def get_database_schema(self) -> Dict[str, Any]:
        """
        Get database schema information
        
        Returns:
            Database schema details
        """
        if not self.engine:
            return {"error": "Database connection not available"}
        
        try:
            inspector = inspect(self.engine)
            
            schema_info = {
                "tables": {},
                "views": {},
                "table_count": 0,
                "total_columns": 0
            }
            
            # Get table information
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                schema_info["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col.get("nullable", True)
                        }
                        for col in columns
                    ],
                    "column_count": len(columns)
                }
                schema_info["total_columns"] += len(columns)
            
            schema_info["table_count"] = len(schema_info["tables"])
            
            # Get view information
            for view_name in inspector.get_view_names():
                try:
                    columns = inspector.get_columns(view_name)
                    schema_info["views"][view_name] = {
                        "columns": [col["name"] for col in columns],
                        "column_count": len(columns)
                    }
                except:
                    schema_info["views"][view_name] = {"columns": [], "column_count": 0}
            
            return schema_info
            
        except Exception as e:
            return {"error": f"Error getting database schema: {e}"}
    
    def execute_custom_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a custom SQL query
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Query results
        """
        if not self.engine:
            return {"error": "Database connection not available"}
        
        try:
            print(f"Executing custom query: {sql_query[:100]}...")
            
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = result.keys()
                rows = result.fetchall()
            
            data = [dict(zip(columns, row)) for row in rows]
            
            return {
                "sql_query": sql_query,
                "data": data,
                "row_count": len(data),
                "columns": list(columns),
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Error executing query: {e}",
                "sql_query": sql_query
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get SQL agent system status
        
        Returns:
            System status information
        """
        return {
            "database_connected": self.engine is not None,
            "sql_agent_available": self.sql_agent is not None,
            "llm_available": self.llm is not None,
            "database_url": self.database_url if self.use_sqlite else "External DB",
            "llm_model": self.llm_model if self.llm else "None",
            "available_kpis": len(self.kpi_definitions),
            "timestamp": datetime.now().isoformat()
        }


def main():
    """
    Demonstration of the SQL agent
    """
    print("Enterprise SQL Agent Demo")
    print("=" * 50)
    
    # Initialize SQL agent
    print("1. Initializing SQL agent...")
    sql_agent = EnterpriseSQLAgent(use_sqlite=True)
    
    # Check system status
    status = sql_agent.get_system_status()
    print("\n2. System Status:")
    print(f"   Database Connected: {'✓' if status['database_connected'] else '✗'}")
    print(f"   SQL Agent Available: {'✓' if status['sql_agent_available'] else '✗'}")
    print(f"   LLM Available: {'✓' if status['llm_available'] else '✗'}")
    print(f"   Available KPIs: {status['available_kpis']}")
    
    # Show available KPIs
    print("\n3. Available KPIs:")
    kpis = sql_agent.get_available_kpis()
    for name, description in kpis.items():
        print(f"   {name}: {description}")
    
    # Execute sample KPI queries
    print("\n4. Executing Sample KPIs:")
    
    sample_kpis = ["sales_by_region", "revenue_by_product_category", "top_products"]
    
    for kpi_name in sample_kpis:
        print(f"\nExecuting KPI: {kpi_name}")
        result = sql_agent.execute_kpi_query(kpi_name)
        
        if "error" in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Description: {result['description']}")
            print(f"   Rows returned: {result['row_count']}")
            print(f"   Columns: {', '.join(result['columns'])}")
            
            if result['data']:
                print(f"   Sample data: {result['data'][0]}")
    
    # Demo natural language queries
    print("\n5. Natural Language Query Demo:")
    
    demo_questions = [
        "What are the total sales by region?",
        "Which product categories generate the most revenue?",
        "Show me the top 5 products by sales",
        "What is the monthly sales trend?"
    ]
    
    for question in demo_questions:
        print(f"\nQuestion: {question}")
        result = sql_agent.query_with_agent(question)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Agent Used: {'✓' if result['agent_used'] else '✗'}")
    
    # Show database schema
    print("\n6. Database Schema:")
    schema = sql_agent.get_database_schema()
    if "error" not in schema:
        print(f"   Tables: {schema['table_count']}")
        print(f"   Views: {len(schema['views'])}")
        print(f"   Total Columns: {schema['total_columns']}")
        
        print(f"\n   Table Details:")
        for table_name, table_info in list(schema['tables'].items())[:3]:
            print(f"     {table_name}: {table_info['column_count']} columns")
    
    print("\n" + "=" * 50)
    print("SQL Agent Demo Complete!")
    
    if not sql_agent.llm:
        print("\nNote: Set OPENAI_API_KEY environment variable for full agent capabilities")
    
    return sql_agent


if __name__ == "__main__":
    agent = main() 