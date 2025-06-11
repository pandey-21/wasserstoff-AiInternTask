import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from typing import List
from .utils import DocumentSnippet, generate_doc_id

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def _chunk_text_into_snippets(doc_id: str, page_num: int, text: str) -> List[DocumentSnippet]:
    """
    A helper function to split a large block of text into paragraph-based snippets.
    Paragraphs are a more semantically coherent unit than fixed-size chunks.
    """
    snippets = []
    # Splitting by double newlines ('\n\n') is a reliable heuristic for paragraph breaks.
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    for para_num, para_text in enumerate(paragraphs):
        snippets.append(DocumentSnippet(
            doc_id=doc_id,
            content=para_text,
            page=page_num,
            paragraph=para_num + 1 # 1-indexed for human-readable citations
        ))
    return snippets

def _process_pdf(file_stream, doc_id: str) -> List[DocumentSnippet]:
    """
    Processes a PDF file using a robust two-step approach:
    1. Tries to extract text directly.
    2. If that fails (indicating a scanned document), falls back to OCR.
    """
    all_snippets = []
    pdf_document = fitz.open(stream=file_stream.read(), filetype="pdf")

    for page_num, page in enumerate(pdf_document):
        page_number_for_citation = page_num + 1
        text = page.get_text().strip()

        # A threshold determines if the page is likely image-based and requires OCR.
        TEXT_LENGTH_THRESHOLD = 100
        if len(text) < TEXT_LENGTH_THRESHOLD:
            try:
                # This is a robust OCR method: render the entire page as a high-DPI image.
                # This works consistently for scanned, skewed, or complex pages.
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR on the generated image.
                ocr_text = pytesseract.image_to_string(img)
                text = ocr_text.strip()
            except Exception as e:
                print(f"ERROR: OCR pipeline failed for page {page_number_for_citation} in {doc_id}: {e}")
                text = "" # Gracefully skip the page on OCR failure

        if text:
            # Once text is obtained (either directly or via OCR), chunk it.
            page_snippets = _chunk_text_into_snippets(doc_id, page_number_for_citation, text)
            all_snippets.extend(page_snippets)
            
    return all_snippets

def process_uploaded_file(uploaded_file) -> List[DocumentSnippet]:
    """
    Main dispatcher function. It identifies the file type and routes it to the
    appropriate processing function.
    """
    file_name = uploaded_file.name.lower()
    doc_id = generate_doc_id(file_name)

    if file_name.endswith('.pdf'):
        return _process_pdf(uploaded_file, doc_id)
    
    elif file_name.endswith(('.txt', '.md')):
        text_content = uploaded_file.read().decode('utf-8', errors='ignore')
        return _chunk_text_into_snippets(doc_id, page_num=1, text=text_content)
    
    elif file_name.endswith(('.png', '.jpg', 'jpeg')):
        image = Image.open(uploaded_file).convert("RGB")
        ocr_text = pytesseract.image_to_string(image)
        return _chunk_text_into_snippets(doc_id, page_num=1, text=ocr_text)
    
    else:
        print(f"WARNING: Unsupported file type '{file_name}'. Skipping.")
        return []
