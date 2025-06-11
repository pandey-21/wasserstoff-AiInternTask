import streamlit as st
import pandas as pd
from core.document_processor import process_uploaded_file
from core.retrieval import VectorStore
from core.generation import get_answer_from_document, synthesize_themes 
from core.utils import generate_doc_id

# --- Page and Session State Setup ---
# Configure the Streamlit page for a clean, wide layout.
st.set_page_config(
    page_title="Wasserstoff AI Intern Task",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables to persist data across user interactions.
# This robust pattern ensures the app doesn't crash on reruns.
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore(collection_name="wasserstoff_task")

# This is the key to preventing duplicate document processing.
if 'processed_doc_ids' not in st.session_state:
    # On first load, sync the in-app state with the database's state.
    # This makes the app stateful across server restarts.
    st.session_state.processed_doc_ids = set(st.session_state.vector_store.get_all_doc_ids())

if 'results' not in st.session_state:
    st.session_state.results = None
if 'themes' not in st.session_state:
    st.session_state.themes = None

# --- Sidebar UI for Document Management ---
with st.sidebar:
    st.header("1. Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Image files",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if st.button("Process and Ingest Documents"):
        if uploaded_files:
            # First, identify which of the uploaded files are truly new.
            new_files_to_process = []
            for file in uploaded_files:
                doc_id = generate_doc_id(file.name.lower())
                # Check against our persistent set of processed IDs.
                if doc_id in st.session_state.processed_doc_ids:
                    st.info(f"'{file.name}' has already been ingested. Skipping.")
                else:
                    new_files_to_process.append((file, doc_id))
            
            # Now, process only the new files, one by one.
            if new_files_to_process:
                with st.spinner(f"Processing {len(new_files_to_process)} new document(s)..."):
                    for file, doc_id in new_files_to_process:
                        try:
                            # Call the main processing pipeline for a single file.
                            snippets = process_uploaded_file(file)
                            if snippets:
                                # Add document chunks to the vector database.
                                st.session_state.vector_store.add_documents(snippets)
                                # IMPORTANT: Update the state only after successful ingestion.
                                st.session_state.processed_doc_ids.add(doc_id)
                                st.success(f"Successfully ingested `{file.name}`.")
                            else:
                                st.warning(f"No content could be extracted from `{file.name}`.")
                        except Exception as e:
                            st.error(f"Failed to process `{file.name}`: {e}")
            else:
                st.success("No new documents to ingest.")
        else:
            st.warning("Please upload at least one document.")

    st.subheader("Knowledge Base")
    doc_ids_list = sorted(list(st.session_state.processed_doc_ids))
    if doc_ids_list:
        st.write(f"**{len(doc_ids_list)}** documents in database:")
        st.text_area("doc_ids", "\n".join(doc_ids_list), height=150, disabled=True)
    else:
        st.info("Knowledge base is empty.")

    # Allows the user to reset the entire application state and database.
    if st.button("Clear Knowledge Base"):
        st.session_state.vector_store.clear_database()
        st.session_state.processed_doc_ids = set()
        st.session_state.results = None
        st.session_state.themes = None
        st.rerun()

# --- Main Page UI for Q&A ---
st.title("📄 Document Research & Theme Identification Chatbot")
st.markdown("Upload documents via the sidebar, then ask a question below.")

st.header("2. Ask a Question")
query = st.text_input("Your question:", placeholder="e.g., What are the main risks identified across all reports?")

# The button is disabled until documents are loaded and a query is entered.
if st.button("Get Answers & Themes", disabled=not query or not st.session_state.processed_doc_ids):
    st.session_state.results = None
    st.session_state.themes = None

    # --- Stage 1: Per-Document Analysis ---
    with st.spinner("Searching and analyzing documents..."):
        all_doc_ids = list(st.session_state.processed_doc_ids)
        answers = []
        progress_bar = st.progress(0, "Starting analysis...")
        for i, doc_id in enumerate(all_doc_ids):
            progress_bar.progress((i + 1) / len(all_doc_ids), f"Analyzing: {doc_id}")
            # Retrieve relevant text chunks for only one document at a time.
            snippets = st.session_state.vector_store.search_in_document(query, doc_id)
            # Generate a cited answer from those chunks.
            answer_obj = get_answer_from_document(query, snippets, doc_id)
            answers.append(answer_obj)
        st.session_state.results = answers

    # --- Stage 2: Cross-Document Synthesis ---
    with st.spinner("Synthesizing themes across all answers..."):
        st.session_state.themes = synthesize_themes(query, st.session_state.results)

# --- Results Display Section ---
# This part of the UI only appears after a query has been processed.
if st.session_state.results:
    st.header("3. Results")
    st.subheader("Individual Document Answers")
    # Format results into a pandas DataFrame for a clean tabular display.
    df_data = [{
        "Document ID": res.doc_id,
        "Extracted Answer": res.extracted_answer,
        "Citation": f"Page {res.source_page}, Para {res.source_paragraph}" if res.source_page else "N/A"
    } for res in st.session_state.results]
    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

if st.session_state.themes is not None:
    st.subheader("Synthesized Themes")
    if not st.session_state.themes:
        st.info("No common themes could be identified from the document answers.")
    else:
        for theme in st.session_state.themes:
            with st.container(border=True):
                st.markdown(f"#### {theme.theme_title}")
                st.markdown(f"**Summary:** {theme.summary}")
                st.markdown(f"**Supporting Documents:** `{', '.join(theme.supporting_docs)}`")