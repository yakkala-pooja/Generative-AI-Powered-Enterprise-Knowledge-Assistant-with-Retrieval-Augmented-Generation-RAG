import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# Other imports
import pandas as pd
import numpy as np
from pathlib import Path
import random

class EnterpriseKnowledgeProcessor:
    """
    Enterprise Knowledge Processing System using LangChain
    
    Processes supply chain SOPs, reports, and customer support logs.
    Creates embeddings and stores them in FAISS for retrieval.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers", openai_api_key: Optional[str] = None):
        """
        Initialize the processor
        
        Args:
            embedding_model: Either "openai" or "sentence-transformers"
            openai_api_key: OpenAI API key if using OpenAI embeddings
        """
        self.embedding_model_type = embedding_model
        self.documents_dir = "enterprise_documents"
        self.vector_store_path = "faiss_index"
        self.metadata_path = "document_metadata.json"
        
        # Initialize embeddings
        if embedding_model == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            print(f"Initialized OpenAI Ada embeddings")
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print(f"Initialized sentence-transformers/all-MiniLM-L6-v2 embeddings")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vector_store = None
        self.document_metadata = {}
        
        # Create directories
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs("processed_data", exist_ok=True)
    
    def generate_sample_documents(self) -> Dict[str, List[str]]:
        """
        Generate sample enterprise documents for demonstration
        
        Returns:
            Dictionary with document types and their file paths
        """
        print("Generating sample enterprise documents...")
        
        documents = {
            "supply_chain_sops": [],
            "reports": [],
            "customer_support_logs": []
        }
        
        # Generate Supply Chain SOPs
        sop_templates = [
            {
                "title": "Warehouse Inventory Management SOP",
                "content": """
STANDARD OPERATING PROCEDURE
Document ID: SOP-WH-001
Version: 2.1
Effective Date: {date}

PURPOSE:
This procedure outlines the standard process for managing warehouse inventory to ensure accurate stock levels, minimize waste, and maintain optimal storage conditions.

SCOPE:
This SOP applies to all warehouse personnel responsible for inventory management, including receiving, storage, picking, and shipping operations.

RESPONSIBILITIES:
- Warehouse Manager: Overall responsibility for inventory accuracy
- Inventory Clerks: Daily cycle counts and stock movements
- Receiving Team: Incoming shipment verification
- Shipping Team: Outbound order fulfillment

PROCEDURE:
1. DAILY INVENTORY CHECKS
   1.1 Perform cycle counts on assigned SKUs
   1.2 Update inventory management system
   1.3 Report discrepancies to supervisor
   1.4 Document all adjustments with proper authorization

2. RECEIVING PROCESS
   2.1 Verify shipment against purchase order
   2.2 Inspect goods for quality and damage
   2.3 Update system with received quantities
   2.4 Place items in designated storage locations

3. STORAGE GUIDELINES
   3.1 Follow FIFO (First In, First Out) methodology
   3.2 Maintain proper temperature and humidity conditions
   3.3 Ensure adequate spacing for safety and accessibility
   3.4 Label all storage locations clearly

4. ORDER FULFILLMENT
   4.1 Pick items based on shipping priority
   4.2 Verify picked quantities against order
   4.3 Package according to shipping requirements
   4.4 Update system with shipped quantities

QUALITY CONTROL:
- Weekly inventory accuracy audits
- Monthly storage condition reviews
- Quarterly process optimization reviews

DOCUMENTATION:
All inventory movements must be recorded in the WMS system within 2 hours of completion.

TRAINING:
New employees must complete inventory management training within 30 days of hire.
"""
            },
            {
                "title": "Supplier Qualification SOP",
                "content": """
STANDARD OPERATING PROCEDURE
Document ID: SOP-SC-002
Version: 1.8
Effective Date: {date}

PURPOSE:
To establish a standardized process for evaluating and qualifying new suppliers to ensure they meet quality, delivery, and compliance requirements.

SCOPE:
This procedure applies to all procurement activities involving new supplier selection and qualification.

PROCEDURE:
1. INITIAL SUPPLIER ASSESSMENT
   1.1 Review supplier capabilities and certifications
   1.2 Evaluate financial stability
   1.3 Assess quality management systems
   1.4 Review references from existing customers

2. QUALIFICATION CRITERIA
   2.1 Quality certifications (ISO 9001, etc.)
   2.2 On-time delivery performance (>95% target)
   2.3 Financial stability (minimum credit rating)
   2.4 Compliance with regulatory requirements

3. SITE AUDIT PROCESS
   3.1 Schedule on-site audit within 30 days
   3.2 Evaluate manufacturing processes
   3.3 Review quality control procedures
   3.4 Assess environmental and safety practices

4. APPROVAL PROCESS
   4.1 Compile audit results and recommendations
   4.2 Present to supplier qualification committee
   4.3 Make go/no-go decision
   4.4 Communicate decision to supplier and internal teams

RISK ASSESSMENT:
Suppliers are categorized as Low, Medium, or High risk based on:
- Geographic location
- Single source dependencies
- Financial stability
- Quality history

CONTINUOUS MONITORING:
Approved suppliers undergo quarterly performance reviews covering:
- Delivery performance
- Quality metrics
- Cost competitiveness
- Innovation capabilities
"""
            },
            {
                "title": "Cold Chain Management SOP",
                "content": """
STANDARD OPERATING PROCEDURE
Document ID: SOP-CC-003
Version: 3.0
Effective Date: {date}

PURPOSE:
To maintain product integrity during cold chain transportation and storage operations.

SCOPE:
Applies to all temperature-sensitive products requiring controlled storage and transportation.

CRITICAL CONTROL POINTS:
1. Receiving temperature verification
2. Storage temperature monitoring
3. Transportation temperature control
4. Delivery temperature confirmation

TEMPERATURE REQUIREMENTS:
- Frozen products: -18°C to -15°C
- Chilled products: 2°C to 8°C
- Ambient sensitive: 15°C to 25°C

MONITORING PROCEDURES:
1. Continuous temperature logging every 15 minutes
2. Immediate alerts for temperature deviations
3. Manual temperature checks every 4 hours
4. Calibration of monitoring equipment monthly

DEVIATION RESPONSE:
1. Immediate investigation of root cause
2. Product quality assessment
3. Customer notification if required
4. Corrective action implementation
5. Documentation of all actions taken

TRAINING REQUIREMENTS:
All personnel handling temperature-sensitive products must complete annual cold chain training.

DOCUMENTATION:
Temperature logs must be maintained for minimum 2 years and available for customer/regulatory audits.
"""
            }
        ]
        
        for i, sop in enumerate(sop_templates):
            filename = f"sop_{i+1}_{sop['title'].lower().replace(' ', '_')}.txt"
            filepath = os.path.join(self.documents_dir, filename)
            
            content = sop['content'].format(
                date=(datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents["supply_chain_sops"].append(filepath)
        
        # Generate Reports
        report_templates = [
            {
                "title": "Monthly Supply Chain Performance Report",
                "content": """
MONTHLY SUPPLY CHAIN PERFORMANCE REPORT
Report Period: {month} {year}
Generated: {date}

EXECUTIVE SUMMARY:
This report provides a comprehensive analysis of supply chain performance metrics for the month of {month} {year}.

KEY PERFORMANCE INDICATORS:
1. Order Fulfillment Rate: 97.3% (Target: 95%)
2. On-Time Delivery: 94.8% (Target: 95%)
3. Inventory Turnover: 8.2x (Target: 8.0x)
4. Supplier Performance Score: 92.1% (Target: 90%)

OPERATIONAL METRICS:
- Total Orders Processed: 15,247
- Average Order Processing Time: 2.3 hours
- Warehouse Utilization: 78%
- Transportation Cost per Unit: $2.43

SUPPLY CHAIN RESILIENCE:
Risk Assessment Summary:
- Geographic concentration risk: Medium
- Supplier dependency risk: Low
- Demand volatility risk: Medium

Mitigation Actions:
1. Diversification of supplier base in Asia-Pacific region
2. Implementation of demand sensing technology
3. Establishment of strategic safety stock levels

COST ANALYSIS:
Total Supply Chain Cost: $2.4M
- Procurement: 65%
- Warehousing: 20%
- Transportation: 12%
- Technology: 3%

Year-over-year cost reduction: 3.2%

QUALITY METRICS:
- Defect Rate: 0.08% (Target: <0.1%)
- Customer Complaints: 12 (Previous month: 18)
- Supplier Quality Score: 94.6%

SUSTAINABILITY METRICS:
- Carbon Footprint Reduction: 5.3% YoY
- Packaging Waste Reduction: 8.1%
- Renewable Energy Usage: 23%

RECOMMENDATIONS:
1. Investigate root causes of on-time delivery misses
2. Optimize warehouse layout to improve utilization
3. Implement predictive analytics for demand forecasting
4. Expand supplier sustainability scorecard program

NEXT MONTH FOCUS AREAS:
- Peak season preparation
- Supplier capacity planning
- Technology system upgrades
"""
            },
            {
                "title": "Quarterly Risk Assessment Report",
                "content": """
QUARTERLY RISK ASSESSMENT REPORT
Q{quarter} {year} - Supply Chain Risk Analysis
Generated: {date}

RISK EXECUTIVE SUMMARY:
Overall supply chain risk level: MEDIUM
Critical risks requiring immediate attention: 2
Risks under monitoring: 8
Risks mitigated this quarter: 3

CRITICAL RISK ANALYSIS:

1. SUPPLIER CONCENTRATION RISK
Risk Level: HIGH
Description: 40% of critical components sourced from single supplier
Impact: Potential production stoppage, revenue loss $500K-1M per day
Probability: Medium (25-50%)
Mitigation Actions:
- Identify and qualify alternative suppliers
- Negotiate supply agreements with backup vendors
- Increase safety stock for critical components
Timeline: 90 days

2. GEOPOLITICAL RISK - ASIA PACIFIC
Risk Level: HIGH
Description: Trade tensions affecting 35% of supplier base
Impact: Tariffs, delivery delays, cost increases
Probability: High (>50%)
Mitigation Actions:
- Diversify supplier geographic footprint
- Establish regional distribution centers
- Develop nearshoring strategy
Timeline: 180 days

OPERATIONAL RISKS:

3. WAREHOUSE CAPACITY CONSTRAINTS
Risk Level: MEDIUM
Description: Operating at 85% capacity during peak periods
Impact: Delayed shipments, increased costs
Mitigation: Expansion planning underway

4. TRANSPORTATION NETWORK DISRUPTION
Risk Level: MEDIUM
Description: Dependency on limited transportation corridors
Impact: Delivery delays, customer satisfaction impact
Mitigation: Multi-modal transportation strategy

5. CYBERSECURITY THREATS
Risk Level: MEDIUM
Description: Increasing sophistication of supply chain cyber attacks
Impact: System downtime, data breach, operational disruption
Mitigation: Enhanced security protocols, employee training

RISK MONITORING FRAMEWORK:
- Weekly risk dashboard updates
- Monthly supplier risk assessments
- Quarterly comprehensive risk reviews
- Annual business continuity testing

KEY RISK INDICATORS (KRIs):
- Supplier financial health scores
- Geopolitical stability indices
- Capacity utilization rates
- Cybersecurity threat levels

SCENARIO PLANNING:
Best Case: All mitigation actions successful, risk levels reduced by 30%
Most Likely: Gradual risk reduction over 6-month period
Worst Case: Multiple risk events materialize simultaneously

INVESTMENT RECOMMENDATIONS:
1. Risk management technology platform: $250K
2. Supplier diversification program: $500K
3. Business continuity planning: $150K
Total recommended investment: $900K
"""
            }
        ]
        
        for i, report in enumerate(report_templates):
            filename = f"report_{i+1}_{report['title'].lower().replace(' ', '_')}.txt"
            filepath = os.path.join(self.documents_dir, filename)
            
            content = report['content'].format(
                month=random.choice(['January', 'February', 'March', 'April', 'May', 'June']),
                year=random.choice([2023, 2024]),
                quarter=random.choice([1, 2, 3, 4]),
                date=datetime.now().strftime("%Y-%m-%d")
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents["reports"].append(filepath)
        
        # Generate Customer Support Logs
        support_scenarios = [
            {
                "title": "Delivery Delay Issue Resolution",
                "content": """
CUSTOMER SUPPORT LOG
Ticket ID: CS-2024-{ticket_id}
Date Created: {date}
Priority: High
Status: Resolved
Customer: Global Manufacturing Corp
Account Manager: Sarah Johnson

ISSUE DESCRIPTION:
Customer reported delayed delivery of critical manufacturing components affecting their production schedule. Original delivery date was {original_date}, actual delivery occurred on {actual_date}.

CUSTOMER IMPACT:
- Production line shutdown for 6 hours
- Estimated revenue impact: $150,000
- Customer satisfaction score impact
- Potential contract review

ROOT CAUSE ANALYSIS:
1. Supplier manufacturing delay due to equipment failure
2. Insufficient safety stock for critical components
3. Communication breakdown between logistics and customer service teams
4. Weather-related transportation delays

RESOLUTION ACTIONS:
1. Immediate Actions (0-24 hours):
   - Expedited remaining order components via air freight
   - Provided hourly delivery updates to customer
   - Escalated to senior management
   - Offered partial credit for delay impact

2. Short-term Actions (1-7 days):
   - Implemented enhanced monitoring for this customer
   - Increased safety stock levels for critical components
   - Revised delivery commitments with buffer time
   - Conducted supplier performance review

3. Long-term Actions (1-4 weeks):
   - Updated supplier agreements with penalty clauses
   - Implemented predictive analytics for demand planning
   - Enhanced communication protocols
   - Invested in transportation management system upgrade

CUSTOMER COMMUNICATION:
{date} 09:00 - Initial notification of delay received
{date} 09:30 - Customer contacted, situation explained
{date} 10:00 - Management escalation, expedited shipping arranged
{date} 12:00 - Hourly updates initiated
{actual_date} 14:00 - Delivery completed, customer notified
{followup_date} - Follow-up call to ensure satisfaction

LESSONS LEARNED:
1. Need for better supplier risk management
2. Importance of proactive customer communication
3. Value of having multiple transportation options
4. Critical nature of safety stock for key customers

PROCESS IMPROVEMENTS:
1. Early warning system for potential delays
2. Automated customer notification system
3. Supplier diversification strategy
4. Enhanced business continuity planning

CUSTOMER FEEDBACK:
"While the delay was unfortunate, we appreciate the transparency and proactive communication throughout the resolution process. The expedited delivery and compensation demonstrated your commitment to our partnership."

Final Satisfaction Score: 7/10 (improved from initial 3/10)
"""
            },
            {
                "title": "Product Quality Issue Investigation",
                "content": """
CUSTOMER SUPPORT LOG
Ticket ID: CS-2024-{ticket_id}
Date Created: {date}
Priority: Critical
Status: Closed
Customer: TechCorp Industries
Quality Manager: David Chen

ISSUE DESCRIPTION:
Customer reported quality defects in received electronic components. Batch QC-{batch_id} showing 15% failure rate during customer's incoming inspection, significantly above acceptable threshold of 2%.

DEFECT DETAILS:
- Component Type: Precision resistors, Model PR-4500
- Batch Size: 10,000 units
- Defective Units: ~1,500
- Defect Types: Resistance value drift, physical damage
- Customer Line Impact: Production halt on two assembly lines

IMMEDIATE RESPONSE:
1. Quality hold placed on remaining inventory from same batch
2. Customer advised to quarantine affected components
3. Replacement units expedited from alternate batch
4. Quality team initiated investigation

INVESTIGATION FINDINGS:
Root Cause Analysis:
1. Primary Cause: Supplier manufacturing process variation
   - Temperature control issue during production week of {production_date}
   - Quality control sampling frequency was insufficient
   
2. Contributing Factors:
   - Incoming inspection process missed early indicators
   - Supplier audit overdue by 3 months
   - Transportation handling contributed to physical damage

SUPPLIER RESPONSE:
Supplier: Precision Components Ltd.
Actions Taken:
1. Full production line recalibration
2. Enhanced quality control procedures
3. 100% inspection for next 3 shipments
4. Process improvement plan submitted
5. Credit issued for defective components

CORRECTIVE ACTIONS:
Immediate (0-48 hours):
- Recalled all components from same production lot
- Expedited replacement with 100% tested units
- Implemented enhanced incoming inspection

Short-term (1-2 weeks):
- Conducted emergency supplier audit
- Revised quality agreement with stricter tolerances
- Updated incoming inspection procedures
- Implemented statistical process control

Long-term (1-3 months):
- Supplier improvement plan implementation
- Alternative supplier qualification initiated
- Enhanced quality management system
- Predictive quality analytics deployment

COST IMPACT:
- Replacement components: $45,000
- Expedited shipping: $8,000
- Customer compensation: $25,000
- Internal investigation costs: $12,000
Total Impact: $90,000

CUSTOMER OUTCOME:
1. Production resumed within 48 hours
2. Quality issue resolved without further defects
3. Relationship maintained through transparent communication
4. Process improvements appreciated by customer

PROCESS IMPROVEMENTS:
1. Enhanced supplier quality monitoring
2. Real-time quality dashboard implementation
3. Automated defect trend analysis
4. Customer quality portal development

PREVENTION MEASURES:
1. Monthly supplier quality reviews
2. Statistical sampling optimization
3. Predictive quality modeling
4. Enhanced transportation packaging

Customer Satisfaction: 8/10
Issue Resolution Time: 72 hours
"""
            }
        ]
        
        for i, log in enumerate(support_scenarios):
            filename = f"support_log_{i+1}_{log['title'].lower().replace(' ', '_')}.txt"
            filepath = os.path.join(self.documents_dir, filename)
            
            base_date = datetime.now() - timedelta(days=random.randint(1, 90))
            content = log['content'].format(
                ticket_id=f"{random.randint(10000, 99999)}",
                date=base_date.strftime("%Y-%m-%d"),
                original_date=(base_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                actual_date=(base_date + timedelta(days=2)).strftime("%Y-%m-%d"),
                followup_date=(base_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                batch_id=f"{random.randint(1000, 9999)}",
                production_date=(base_date - timedelta(days=10)).strftime("%Y-%m-%d")
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            documents["customer_support_logs"].append(filepath)
        
        print(f"Generated {len(documents['supply_chain_sops'])} SOPs")
        print(f"Generated {len(documents['reports'])} reports")
        print(f"Generated {len(documents['customer_support_logs'])} support logs")
        
        return documents
    
    def load_documents(self, document_paths: Optional[List[str]] = None) -> List[Document]:
        """
        Load documents from specified paths or from the documents directory
        
        Args:
            document_paths: List of specific document paths to load
            
        Returns:
            List of LangChain Document objects
        """
        print("Loading documents...")
        
        if document_paths is None:
            # Load all documents from the documents directory
            loader = DirectoryLoader(
                self.documents_dir,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            documents = loader.load()
        else:
            # Load specific documents
            documents = []
            for path in document_paths:
                if os.path.exists(path):
                    loader = TextLoader(path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
        
        # Add metadata
        for doc in documents:
            filename = os.path.basename(doc.metadata['source'])
            doc.metadata.update({
                'filename': filename,
                'document_type': self._classify_document_type(filename),
                'processed_date': datetime.now().isoformat(),
                'chunk_count': 0  # Will be updated during chunking
            })
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def _classify_document_type(self, filename: str) -> str:
        """Classify document type based on filename"""
        if 'sop' in filename.lower():
            return 'standard_operating_procedure'
        elif 'report' in filename.lower():
            return 'report'
        elif 'support' in filename.lower() or 'log' in filename.lower():
            return 'customer_support_log'
        else:
            return 'unknown'
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        print("Chunking documents...")
        
        chunked_docs = []
        chunk_id = 0
        
        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_documents([doc])
            
            # Update metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_document': doc.metadata['filename'],
                    'chunk_size': len(chunk.page_content)
                })
                chunk_id += 1
            
            chunked_docs.extend(chunks)
            
            # Update original document metadata
            doc.metadata['chunk_count'] = len(chunks)
        
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from chunked documents
        
        Args:
            documents: List of chunked Document objects
            
        Returns:
            FAISS vector store
        """
        print("Creating embeddings and building FAISS index...")
        
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        # Create vector store
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        self.vector_store = vector_store
        
        print(f"Created FAISS index with {len(documents)} document chunks")
        return vector_store
    
    def save_vector_store(self, vector_store: FAISS) -> None:
        """
        Save the FAISS vector store to disk
        
        Args:
            vector_store: FAISS vector store to save
        """
        print("Saving vector store to disk...")
        
        # Save FAISS index
        vector_store.save_local(self.vector_store_path)
        
        # Save metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'embedding_model': self.embedding_model_type,
            'total_chunks': vector_store.index.ntotal,
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Vector store saved to {self.vector_store_path}")
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load existing vector store from disk
        
        Returns:
            FAISS vector store if exists, None otherwise
        """
        if os.path.exists(self.vector_store_path):
            print("Loading existing vector store...")
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True  # Only for trusted local files
            )
            self.vector_store = vector_store
            return vector_store
        return None
    
    def search_documents(self, query: str, k: int = 5, score_threshold: float = 0.85) -> List[Tuple[Document, float]]:
        """
        Search for relevant documents using similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (higher = less strict)
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please process documents first.")
        
        print(f"Searching for: '{query}'")
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Filter by score threshold (higher scores = less similar)
        filtered_results = [(doc, score) for doc, score in results if score <= score_threshold]
        
        if not filtered_results:
            # If no results, return top 2 regardless of score
            filtered_results = results[:2]
        
        print(f"Found {len(filtered_results)} relevant documents")
        return filtered_results
    
    def process_all_documents(self, regenerate: bool = False) -> Dict:
        """
        Complete pipeline: generate documents, chunk, embed, and store
        
        Args:
            regenerate: Whether to regenerate sample documents
            
        Returns:
            Processing statistics
        """
        print("Starting complete document processing pipeline...")
        
        start_time = datetime.now()
        
        # Check if vector store already exists
        if not regenerate and os.path.exists(self.vector_store_path):
            print("Existing vector store found. Loading...")
            self.load_vector_store()
            return {
                'status': 'loaded_existing',
                'vector_store_path': self.vector_store_path,
                'total_chunks': self.vector_store.index.ntotal if self.vector_store else 0
            }
        
        # Step 1: Generate sample documents
        if regenerate or not os.listdir(self.documents_dir):
            document_info = self.generate_sample_documents()
        
        # Step 2: Load documents
        documents = self.load_documents()
        
        # Step 3: Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        # Step 4: Create embeddings and vector store
        vector_store = self.create_vector_store(chunked_documents)
        
        # Step 5: Save vector store
        self.save_vector_store(vector_store)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate statistics
        stats = {
            'status': 'completed',
            'processing_time_seconds': processing_time,
            'total_documents': len(documents),
            'total_chunks': len(chunked_documents),
            'embedding_model': self.embedding_model_type,
            'vector_store_path': self.vector_store_path,
            'documents_by_type': {},
            'avg_chunk_size': np.mean([len(doc.page_content) for doc in chunked_documents]),
            'chunk_size_distribution': {
                'min': min([len(doc.page_content) for doc in chunked_documents]),
                'max': max([len(doc.page_content) for doc in chunked_documents]),
                'std': np.std([len(doc.page_content) for doc in chunked_documents])
            }
        }
        
        # Count documents by type
        for doc in documents:
            doc_type = doc.metadata['document_type']
            stats['documents_by_type'][doc_type] = stats['documents_by_type'].get(doc_type, 0) + 1
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        return stats
    
    def get_document_statistics(self) -> Dict:
        """
        Get detailed statistics about processed documents
        
        Returns:
            Dictionary with comprehensive statistics
        """
        if self.vector_store is None:
            return {"error": "No vector store loaded"}
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            'vector_store_stats': {
                'total_vectors': self.vector_store.index.ntotal,
                'dimension': self.vector_store.index.d,
                'embedding_model': self.embedding_model_type
            },
            'processing_metadata': metadata,
            'document_directory': self.documents_dir,
            'vector_store_path': self.vector_store_path
        }
    
    def demo_search_scenarios(self) -> Dict:
        """
        Demonstrate search capabilities with various queries
        
        Returns:
            Dictionary with search results for different scenarios
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        demo_queries = [
            "warehouse inventory management procedures",
            "supplier qualification process",
            "cold chain temperature requirements",
            "delivery delay resolution",
            "quality defect investigation",
            "risk assessment methodology",
            "customer satisfaction improvement"
        ]
        
        demo_results = {}
        
        for query in demo_queries:
            results = self.search_documents(query, k=3)
            demo_results[query] = [
                {
                    'content': doc.page_content[:200] + "...",
                    'score': float(score),
                    'document_type': doc.metadata.get('document_type', 'unknown'),
                    'filename': doc.metadata.get('filename', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', -1)
                }
                for doc, score in results
            ]
        
        return demo_results


def main():
    """
    Main function to demonstrate the enterprise knowledge processor
    """
    print("Enterprise Knowledge Processing System")
    print("=" * 50)
    
    # Initialize processor with sentence-transformers (no API key required)
    processor = EnterpriseKnowledgeProcessor(embedding_model="sentence-transformers")
    
    # Process all documents
    print("\nProcessing enterprise documents...")
    stats = processor.process_all_documents(regenerate=True)
    
    # Display statistics
    print("\nProcessing Statistics:")
    print(f"Status: {stats['status']}")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Chunks: {stats['total_chunks']}")
    print(f"Processing Time: {stats['processing_time_seconds']:.2f} seconds")
    print(f"Average Chunk Size: {stats['avg_chunk_size']:.0f} characters")
    
    print("\nDocuments by Type:")
    for doc_type, count in stats['documents_by_type'].items():
        print(f"  {doc_type}: {count}")
    
    # Demonstrate search capabilities
    print("\nDemonstrating Search Capabilities:")
    demo_results = processor.demo_search_scenarios()
    
    for query, results in list(demo_results.items())[:3]:  # Show first 3 queries
        print(f"\nQuery: '{query}'")
        for i, result in enumerate(results[:2], 1):  # Show top 2 results
            print(f"  Result {i}:")
            print(f"    Document: {result['filename']}")
            print(f"    Type: {result['document_type']}")
            print(f"    Score: {result['score']:.3f}")
            print(f"    Content: {result['content']}")
    
    # Get system statistics
    system_stats = processor.get_document_statistics()
    print(f"\nVector Store Info:")
    print(f"  Total Vectors: {system_stats['vector_store_stats']['total_vectors']:,}")
    print(f"  Vector Dimension: {system_stats['vector_store_stats']['dimension']}")
    print(f"  Embedding Model: {system_stats['vector_store_stats']['embedding_model']}")
    
    print("\nSystem ready for queries!")
    return processor


if __name__ == "__main__":
    processor = main() 