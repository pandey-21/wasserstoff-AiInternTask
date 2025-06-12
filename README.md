# AI-Powered Document Research & Theme Identification Chatbot

This project is a submission for the Wasserstoff AI Software Intern qualification task. It's a fully functional web application designed to act as an intelligent research assistant, capable of ingesting a large corpus of documents, providing precisely cited answers, and synthesizing high-level thematic insights.

---

## 🚀 Live Application

The application has been deployed and is publicly accessible.

**You can test the live version here:**

### [https://wasserstoff--aiinterntaskstreamlit.app/](https://wasserstoff--aiinterntask.streamlit.app/)

---

## 🎬 Demo Video

A brief video walkthrough demonstrating the core features and workflow of the application is available here:

**[Link to Your Demo Video on YouTube/Loom]**

---

## ✨ Key Features

*   **Multi-Format Document Ingestion**: Supports various document types, including text-based PDFs, scanned PDFs, plain text files, and images (`.png`, `.jpg`).
*   **Robust OCR for Scanned PDFs**: For image-based PDFs, the system renders each page as a high-resolution image and uses an OCR engine to accurately extract text.
*   **Advanced Two-Stage RAG**:
    1.  **Per-Document Analysis**: Extracts answers with precise citations (page and paragraph) from each document individually.
    2.  **Thematic Synthesis**: Analyzes the individual answers to identify and summarize overarching themes across the entire document set.
*   **Interactive Web Interface**: A clean and intuitive UI built with Streamlit allows for easy document upload and querying.
*   **Idempotent & Robust**: The system intelligently handles duplicate file uploads and is resilient against common failures in LLM responses and document processing.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Backend & Core Logic**: Python
*   **LLM Provider**: Groq (for high-speed Llama 3 inference)
*   **Vector Database**: ChromaDB (with its built-in embedding model)
*   **Document Processing**: PyMuPDF & Tesseract
*   **Data Validation**: Pydantic

## ⚙️ Running the Project Locally

### Prerequisites
- Python 3.9+
- Tesseract-OCR installed and available in your system's PATH.
- A Groq API key.

### Setup Instructions
1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The project dependencies are listed in `backend/requirements.txt`.
    ```bash
    pip install -r backend/requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a `.env` file inside the `backend/` directory and add your Groq API key:
    ```
    GROQ_API_KEY="gsk_..."
    ```

5.  **Run the Application:**
    ```bash
    streamlit run backend/app/main.py
    ```
    The application will be available at `http://localhost:8501`.
