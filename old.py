import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import traceback
import sys
from functools import lru_cache
import asyncio
import inspect
import PyPDF2
import io

# LangChain and related imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
from typing import List, Dict, Any, Optional, Union

# LlamaParse and OpenAI imports
from openai import OpenAI
from llama_parse import LlamaParse  # Make sure this is your actual import path

# Load environment variables
load_dotenv()

# Custom formatter for logging
class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.request_id = getattr(record, 'request_id', '-')
        return super().format(record)

# Logging configuration
log_formatter = CustomFormatter('%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s')
log_file = 'app.log'
log_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
log_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler])

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# Constants
TOP_K = 6
OUTPUT_FOLDER = 'parsed_pdfs'
FAISS_INDEX_FOLDER = 'faiss_index'
ALLOWED_EXTENSIONS = {'pdf'}

# Create necessary directories
for folder in [OUTPUT_FOLDER, FAISS_INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Environment variable handling
def get_required_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        logger.critical(f"{var_name} not found in environment variables")
        raise ValueError(f"{var_name} not found in environment variables")
    return value

# Initialize clients
openai_api_key = get_required_env_var("OPENAI_API_KEY")
llama_cloud_api_key = get_required_env_var("LLAMA_CLOUD_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Initialize LlamaParse
parser = LlamaParse(
    api_key=llama_cloud_api_key,
    api_result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
    num_workers=4,
    verbose=True,
    language="en"
)

# Initialize ChatOpenAI with caching
@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",  # Using the latest GPT-4 model
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=openai_api_key,
    )
    
import subprocess
import sys

def ensure_dependencies():
    """Ensure all necessary Python packages are installed."""
    required_packages = {
        "pypdf2": "PyPDF2",
        "openai": "openai",
        "langchain": "langchain",
        "langchain_openai": "langchain-openai",
        "langchain_community": "langchain-community",
        "faiss": "faiss-cpu",
        "streamlit": "streamlit",
        "streamlit_chat": "streamlit-chat",
        "python-dotenv": "python-dotenv"
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
       # st.warning(f"Installing missing dependencies: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                #st.success(f"Successfully installed {package}")
            except Exception as e:
                st.error(f"Failed to install {package}: {str(e)}")
                return False
    
    return True

# Pydantic models
class GapItem(BaseModel):
    description: str = Field(description="Description of the gap between RFP and Response")
    severity: Literal["Low", "Medium", "High"] = Field(description="Severity of the gap")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "Missing technical specifications",
                    "severity": "High"
                }
            ]
        }
    }

class GapAnalysis(BaseModel):
    summary: str = Field(description="Brief summary of the overall gap analysis")
    gaps: List[GapItem] = Field(description="List of identified gaps")
    suggestions: List[str] = Field(description="List of suggestions to address the gaps")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Several gaps identified between RFP and Response",
                    "gaps": [
                        {
                            "description": "Missing technical specifications",
                            "severity": "High"
                        }
                    ],
                    "suggestions": [
                        "Include detailed technical specifications"
                    ]
                }
            ]
        }
    }

# Custom exceptions
class DocumentProcessingError(Exception):
    pass

class RetrieverError(Exception):
    pass

# FAISS operations
class FAISSOperations:
    @staticmethod
    def clear_index(collection_name: str) -> None:
        try:
            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                logger.info(f"Cleared FAISS index for collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error clearing FAISS index for collection {collection_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to clear FAISS index: {str(e)}")

    @staticmethod
    def create_index(documents: List[Document], collection_name: str) -> FAISS:
        try:
            # Validate documents
            if not documents or len(documents) == 0:
                raise ValueError("No documents provided for creating index")
            
            # Ensure all documents have content
            valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
            
            if not valid_documents:
                raise ValueError("No valid document content provided for creating index")
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            texts = [doc.page_content for doc in valid_documents]
            metadatas = [doc.metadata for doc in valid_documents]
            
            # Create vectorstore
            try:
                vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            except Exception as e:
                logger.error(f"Error creating FAISS vectorstore: {str(e)}")
                raise
            
            # Save vectorstore
            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)

            logger.info(f"Created new FAISS index for collection: {collection_name}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating FAISS index for collection {collection_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to create FAISS index: {str(e)}")
# Document Processing
class DocumentProcessor:
    @staticmethod
    def extract_text_with_pypdf2(file_path: str) -> List[str]:
        """Fallback method for extracting text from PDFs using PyPDF2."""
        try:
            text_pages = []
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text and text.strip():
                        text_pages.append(text)
            
            if not text_pages:
                raise ValueError("No text content could be extracted from the PDF")
            
            return text_pages
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            raise ValueError(f"PyPDF2 extraction failed: {str(e)}")

    @staticmethod
    def parse_pdf(file_path: str, output_name: str) -> str:
        try:
            FAISSOperations.clear_index(output_name)
            
            # First attempt: Use LlamaParse
            try:
                logger.info(f"Attempting to parse {file_path} with LlamaParse")
                result = parser.load_data(file_path)
                
                # Verify we got results
                if result and len(result) > 0 and any(page.text.strip() for page in result):
                    logger.info(f"Successfully extracted content with LlamaParse from {file_path}")
                    # Process with LlamaParse results
                    documents = [
                        Document(page_content=page.text, metadata={"source": output_name, "page": i, "method": "llama_parse"})
                        for i, page in enumerate(result) if page.text.strip()
                    ]
                else:
                    logger.warning(f"LlamaParse returned empty results for {file_path}, trying fallback method")
                    raise ValueError("No content extracted by LlamaParse")
                    
            except Exception as parse_error:
                logger.warning(f"LlamaParse failed: {str(parse_error)}. Trying fallback method...")
                
                # Fallback: Use PyPDF2
                try:
                    text_pages = DocumentProcessor.extract_text_with_pypdf2(file_path)
                    logger.info(f"Successfully extracted content with PyPDF2 from {file_path}")
                    
                    documents = [
                        Document(page_content=page_text, metadata={"source": output_name, "page": i, "method": "pypdf2"})
                        for i, page_text in enumerate(text_pages)
                    ]
                except Exception as fallback_error:
                    logger.error(f"All extraction methods failed for {file_path}")
                    raise DocumentProcessingError(f"All PDF extraction methods failed: {str(fallback_error)}")
            
            # Write extracted content to markdown file
            output_path = os.path.join(OUTPUT_FOLDER, f"{output_name}.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    f.write(f"## Page {doc.metadata['page']}\n\n")
                    f.write(doc.page_content)
                    f.write("\n\n---\n\n")
            
            # Create the index
            if documents:
                FAISSOperations.create_index(documents, output_name)
                return f"Successfully processed {output_name} with {len(documents)} pages"
            else:
                raise DocumentProcessingError(f"No valid content extracted from {file_path}")
                
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise DocumentProcessingError(f"Failed to parse PDF: {str(e)}")

# Document Retrieval
class DocumentRetriever:
    @staticmethod
    def initialize_retriever(collection_name: str) -> Optional[Any]:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            index_path = os.path.join(FAISS_INDEX_FOLDER, collection_name)

            if not os.path.exists(index_path):
                logger.warning(f"No FAISS index found for collection: {collection_name}")
                return None

            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
        except Exception as e:
            logger.error(f"Error initializing retriever for {collection_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RetrieverError(f"Failed to initialize retriever: {str(e)}")

    @staticmethod
    def retrieve_documents(query: str, retriever: Any) -> str:
        try:
            docs = retriever.invoke(query)
            return "\n\n".join([
                f"**Document {i+1}:**\n{doc.page_content}" 
                for i, doc in enumerate(docs)
            ])
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise RetrieverError(f"Failed to retrieve documents: {str(e)}")

# Analysis
class Analyzer:
    @staticmethod
    def analyze_gap(context: str) -> str:
        try:
            llm = get_llm()
            gap_analysis_parser = PydanticOutputParser(pydantic_object=GapAnalysis)

            gap_analysis_prompt = PromptTemplate(
                template="Analyze the gap between the RFP requirements and the Response based on the following context:\n\n{context}\n\n{format_instructions}\n",
                input_variables=["context"],
                partial_variables={"format_instructions": gap_analysis_parser.get_format_instructions()},
            )

            output = llm.invoke(gap_analysis_prompt.format(context=context))
            parsed_output = gap_analysis_parser.parse(output.content)

            result = parsed_output.model_dump()

            # Generate table for gaps with severity highlighting using Markdown
            gaps_table = """
| Description | Severity |
|-------------|----------|
"""
            for gap in result['gaps']:
                severity_color = {
                    "High": "#FF0000",  # Red
                    "Medium": "#FFA500",  # Orange
                    "Low": "#00FF00"  # Green
                }.get(gap['severity'], "#000000")  # Default to black

                # Using Markdown to highlight severity
                gaps_table += f"| {gap['description']} | {gap['severity']} |\n"

            return f"""
Summary: {result['summary']}

### Gaps:
{gaps_table}

### Suggestions:
{chr(10).join([f"- {suggestion}" for suggestion in result['suggestions']])}
"""
        except Exception as e:
            logger.error(f"Error during gap analysis: {str(e)}")
            raise ValueError(f"Failed to analyze gap: {str(e)}")
    @staticmethod
    def generate_insights(context: str) -> str:
        try:
            llm = get_llm()
            insight_prompt = f"""
            Based on the following documents:

            {context}

            Please provide a structured report with the following sections:

            1. Executive Summary:
               - Provide a concise overview of the main points from both the RFP and the Response.

            2. RFP Requirements Checklist:
               - List the critical requirements from the RFP.
               - For each requirement, clearly indicate whether it is addressed in the Response (Addressed/Not Addressed).
               - Provide a brief explanation for each status. If a requirement is addressed, explain how it is addressed. If not addressed, explain why it is missing.
               - Do not use "Partially Addressed" - clearly determine if the requirement is fully addressed or not addressed.

            3. Key Insights:
               - Bullet point the most critical insights derived from comparing the RFP and the Response.
               - For each insight, provide a brief explanation of its significance.

            4. Trends and Patterns:
               - Identify and explain any common themes or patterns across both documents.

            5. Comparative Analysis:
               - Highlight notable differences between the RFP requirements and the Response.
               - Identify any areas where the Response exceeds RFP expectations.
            """

            insights = llm.invoke(insight_prompt)
            return insights.content
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise ValueError(f"Failed to generate insights: {str(e)}")

# Report Formatting
class ReportFormatter:
    @staticmethod
    def format_report(raw_data: str) -> str:
        try:
            formatted_report = f"""
            # RFP and Response Analysis Report

            ## Part 1: Gap Analysis
            {raw_data.split('## Part 1:')[1].split('## Part 2:')[0]}

            ## Part 2: Detailed Insights
            {raw_data.split('## Part 2:')[1]}
            """
            return formatted_report
        except Exception as e:
            logger.error(f"Error formatting report: {str(e)}")
            return f"Error formatting report: {str(e)}"

# Agent Tools
class AgentTools:
    @staticmethod
    @tool
    def retrieve_rfp_documents(query: str) -> str:
        """Retrieve relevant RFP documents using the query."""
        try:
            retriever = DocumentRetriever.initialize_retriever("rfp_parsed")
            if not retriever:
                return "Error: RFP documents not processed yet."

            return DocumentRetriever.retrieve_documents(query, retriever)
        except Exception as e:
            logger.error(f"Error retrieving RFP documents: {str(e)}")
            return f"Error retrieving RFP documents: {str(e)}"

    @staticmethod
    @tool
    def retrieve_response_documents(query: str) -> str:
        """Retrieve relevant Response documents using the query."""
        try:
            retriever = DocumentRetriever.initialize_retriever("response_parsed")
            if not retriever:
                return "Error: Response documents not processed yet."

            return DocumentRetriever.retrieve_documents(query, retriever)
        except Exception as e:
            logger.error(f"Error retrieving Response documents: {str(e)}")
            return f"Error retrieving Response documents: {str(e)}"

    @staticmethod
    def setup_agent():
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )

            tools = [
                Tool(
                    name="Retrieve RFP Documents",
                    func=AgentTools.retrieve_rfp_documents,
                    description="Retrieve relevant RFP documents using the query."
                ),
                Tool(
                    name="Retrieve Response Documents",
                    func=AgentTools.retrieve_response_documents,
                    description="Retrieve relevant Response documents using the query."
                ),
                Tool(
                    name="Analyze Gap",
                    func=Analyzer.analyze_gap,
                    description="Analyze gaps between RFP requirements and Response."
                ),
                Tool(
                    name="Generate Insights",
                    func=Analyzer.generate_insights,
                    description="Generate detailed insights from documents."
                )
            ]

            return initialize_agent(
                tools,
                get_llm(),
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise ValueError(f"Failed to setup agent: {str(e)}")

# Streamlit UI
def main():
    st.set_page_config(page_title="RFP and Response Analyzer", layout="wide")
    
    # Initialize session state
    if 'rfp_processed' not in st.session_state:
        st.session_state.rfp_processed = False
    if 'response_processed' not in st.session_state:
        st.session_state.response_processed = False
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_content' not in st.session_state:
        st.session_state.report_content = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Home", "Process Documents", "Generate Report", "Chat"])
    
    if page == "Home":
        show_home()
    elif page == "Process Documents":
        show_process_page()
    elif page == "Generate Report":
        show_report_page()
    elif page == "Chat":
        show_chat_page()

def show_home():
    st.title("RFP and Response Analyzer")
    st.markdown("""
    This application helps analyze RFP (Request for Proposal) documents and their corresponding responses.
    It provides gap analysis, detailed insights, and formatted reports.
    """)
    st.image("https://via.placeholder.com/800x400.png?text=RFP+Response+Analyzer")  # Placeholder image

def show_process_page():
    st.title("Process Documents")
    
    uploaded_rfp = st.file_uploader("Upload RFP PDF", type=["pdf"])
    uploaded_response = st.file_uploader("Upload Response PDF", type=["pdf"])
    
    if uploaded_rfp and uploaded_response:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily with unique names to avoid conflicts
                    import uuid
                    rfp_path = f"temp_rfp_{uuid.uuid4()}.pdf"
                    response_path = f"temp_response_{uuid.uuid4()}.pdf"
                    
                    try:
                        with open(rfp_path, "wb") as f:
                            f.write(uploaded_rfp.getbuffer())
                        with open(response_path, "wb") as f:
                            f.write(uploaded_response.getbuffer())
                        
                        # Process RFP
                        with st.status("Processing RFP document..."):
                            processor = DocumentProcessor()
                            rfp_result = processor.parse_pdf(rfp_path, "rfp_parsed")
                            st.session_state.rfp_processed = True
                            st.success("RFP processed successfully!")
                        
                        # Process Response separately
                        with st.status("Processing Response document..."):
                            response_result = processor.parse_pdf(response_path, "response_parsed")
                            st.session_state.response_processed = True
                            st.success("Response processed successfully!")
                        
                        st.success("Both documents processed successfully!")
                        
                    except DocumentProcessingError as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(f"Document processing error: {str(e)}")
                        logger.debug(traceback.format_exc())
                    except Exception as e:
                        st.error(f"Unexpected error: {str(e)}")
                        logger.error(f"Unexpected error processing documents: {str(e)}")
                        logger.debug(traceback.format_exc())
                    
                finally:
                    # Clean up temporary files
                    for path in [rfp_path, response_path]:
                        if os.path.exists(path):
                            try:
                                os.remove(path)
                            except Exception as e:
                                logger.warning(f"Failed to remove temporary file {path}: {str(e)}")
                
                # Display processing status
                if st.session_state.rfp_processed and st.session_state.response_processed:
                    st.success("✅ Both documents processed successfully. You can now generate a report.")
                elif st.session_state.rfp_processed:
                    st.warning("⚠️ Only RFP document was processed successfully. Please try uploading the Response document again.")
                elif st.session_state.response_processed:
                    st.warning("⚠️ Only Response document was processed successfully. Please try uploading the RFP document again.")
                else:
                    st.error("❌ Document processing failed. Please try again or check the logs for more details.")

def show_report_page():
    st.title("Generate Report")
    
    if not st.session_state.rfp_processed or not st.session_state.response_processed:
        st.error("Please process both RFP and Response documents first.")
        return
    
    if not st.session_state.report_generated or not st.session_state.report_content:
        if st.button("Generate Analysis Report"):
            try:
                retriever = DocumentRetriever()
                rfp_retriever = retriever.initialize_retriever("rfp_parsed")
                response_retriever = retriever.initialize_retriever("response_parsed")
                
                rfp_content = DocumentRetriever.retrieve_documents("Retrieve all relevant RFP content.", rfp_retriever)
                response_content = DocumentRetriever.retrieve_documents("Retrieve all relevant Response content.", response_retriever)
                
                analyzer = Analyzer()
                raw_analysis = analyzer.analyze_gap(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")
                raw_insights = analyzer.generate_insights(f"RFP Content:\n{rfp_content}\n\nResponse Content:\n{response_content}")
                
                raw_report = f"""
                # RFP and Response Analysis Report

                ## Part 1: Gap Analysis
                {raw_analysis}

                ## Part 2: Detailed Insights
                {raw_insights}
                """

                formatter = ReportFormatter()
                formatted_report = formatter.format_report(raw_report)
                
                st.session_state.report_generated = True
                st.session_state.report_content = formatted_report
                
                st.markdown(formatted_report)
                st.download_button(
                    "Download Report",
                    formatted_report,
                    file_name="rfp_response_analysis_report.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                logger.debug(traceback.format_exc())
                st.error(f"Error generating report: {str(e)}")
    else:
        st.markdown(st.session_state.report_content)
        st.download_button(
            "Download Report",
            st.session_state.report_content,
            file_name="rfp_response_analysis_report.txt",
            mime="text/plain"
        )

def show_chat_page():
    st.title("Chat with Analyzer")
    
    if not st.session_state.rfp_processed or not st.session_state.response_processed:
        st.error("Please process both RFP and Response documents first.")
        return
    
    if not st.session_state.report_generated:
        st.error("Please generate the analysis report first.")
        return
    
    # Display chat messages from history
    for i in range(len(st.session_state.chat_history)):
        message(st.session_state.chat_history[i][0], is_user=st.session_state.chat_history[i][1], key=str(i))
    
    # User input
    query = st.text_input("Enter your query:", key="input")
    
    if query and st.button("Send"):
        try:
            agent = AgentTools.setup_agent()
            if not agent:
                raise ValueError("Failed to initialize the agent")
            
            # Retrieve relevant documents
            rfp_docs = AgentTools.retrieve_rfp_documents(query)
            response_docs = AgentTools.retrieve_response_documents(query)
            
            enhanced_query = f"""
            Considering the following document contents:

            RFP Documents:
            {rfp_docs}

            Response Documents:
            {response_docs}

            Please answer the following query:
            {query}
            """
            
            result = agent.run(input=enhanced_query)
            
            # Store the chat history
            st.session_state.chat_history.append((query, True))
            st.session_state.chat_history.append((result, False))
            
            # Rerun the script to update the UI
            st.experimental_rerun()
            
        except Exception as e:
            logger.error(f"Error during chat execution: {str(e)}")
            logger.debug(traceback.format_exc())
            st.error(f"Error during chat: {str(e)}")

def main():
    st.set_page_config(page_title="RFP and Response Analyzer", layout="wide")
    
    # Check for dependencies
    if not ensure_dependencies():
        st.error("Failed to install required dependencies. Please check the logs and try again.")
        return
    
    # Debug information (only in development)
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
    if show_debug:
        st.sidebar.info("Debug mode enabled")
        st.sidebar.text(f"Python version: {sys.version}")
        st.sidebar.text(f"Working directory: {os.getcwd()}")
        st.sidebar.text(f"FAISS index directory exists: {os.path.exists(FAISS_INDEX_FOLDER)}")
        st.sidebar.text(f"Output directory exists: {os.path.exists(OUTPUT_FOLDER)}")
    
    # Initialize session state
    if 'rfp_processed' not in st.session_state:
        st.session_state.rfp_processed = False
    if 'response_processed' not in st.session_state:
        st.session_state.response_processed = False
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'report_content' not in st.session_state:
        st.session_state.report_content = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Home", "Process Documents", "Generate Report", "Chat"])
    
    if page == "Home":
        show_home()
    elif page == "Process Documents":
        show_process_page()
    elif page == "Generate Report":
        show_report_page()
    elif page == "Chat":
        show_chat_page()

if __name__ == "__main__":
    try:
        # Ensure all required directories exist
        for directory in [OUTPUT_FOLDER, FAISS_INDEX_FOLDER]:
            os.makedirs(directory, exist_ok=True)

        # Validate environment variables
        required_vars = ["OPENAI_API_KEY", "LLAMA_CLOUD_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            st.info("Please create a .env file with the following variables: " + ", ".join(required_vars))
        else:
            # Run the Streamlit app
            main()
    except Exception as e:
        st.error(f"Failed to start application: {str(e)}")
        logger.critical(f"Failed to start application: {str(e)}")
        logger.debug(traceback.format_exc())