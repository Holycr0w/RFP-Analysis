# RFP-Response Analyzer

RFP-Response Analyzer is a Flask-based web application that uses AI to analyze and compare Request for Proposal (RFP) documents with their corresponding responses. It leverages OpenAI's language models and vector embeddings to provide insights, gap analysis, and interactive chat functionality.


## Features

- PDF parsing and indexing using LlamaParse and FAISS
- Gap analysis between RFP requirements and responses
- Insight generation and structured report creation
- Interactive chat interface for querying document contents
- Secure handling of API keys and environment variables

## Setup

1. Clone the repository:

    cd rfp-response-analyzer
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    Create a `.env` file in the project root and add the following:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
    ```

4. Run the application:

    ```bash
    python main.py
    ```

## Usage

1. Access the web interface at `http://localhost:5001`
2. Upload RFP and Response PDF documents
3. Process the documents to create FAISS indexes
4. Generate a structured report comparing the RFP and Response
5. Use the chat interface to ask questions about the documents

## Project Structure

- `main.py`: Main application file containing Flask routes and core functionality
- `requirements.txt`: List of Python dependencies
- `templates/`: Directory for HTML templates (index.html should be added here)
- `parsed_pdfs/`: Output folder for parsed PDF content
- `faiss_index/`: Folder for storing FAISS indexes





- OpenAI for providing the language models and embeddings
- LlamaParse for PDF parsing capabilities
- FAISS for efficient similarity search
