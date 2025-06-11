import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from .utils import DocumentAnswer, Theme

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set!")
# Initialize a single Groq client to be reused for all API calls.
groq_client = Groq(api_key=api_key)
# Define models to use: a fast one for extraction, a powerful one for synthesis.
EXTRACTION_MODEL = "llama3-8b-8192"
SYNTHESIS_MODEL = "llama3-70b-8192"

def get_answer_from_document(query: str, snippets: List[Dict[str, Any]], doc_id: str) -> DocumentAnswer:
    if not snippets:
        return DocumentAnswer(doc_id=doc_id)
# Format the context with XML-like tags to clearly separate snippets for the LLM.
    context = "\n\n".join([
        f"<source page='{s['metadata'].get('page', 'N/A')}' paragraph='{s['metadata'].get('paragraph', 'N/A')}'>\n{s['content']}\n</source>"
        for s in snippets
    ])
# This prompt is engineered to be extremely strict, forcing the LLM to synthesize
   # a single JSON object and preventing it from generating multiple answers.
    prompt = f"""
    You are a precise data extraction bot. Your task is to provide a single, consolidated answer to the user's question based on all the provided context snippets.

    Follow these rules STRICTLY:
    1.  Read all the <source> snippets provided in the context.
    2.  Synthesize them to form the BEST POSSIBLE and MOST COMPREHENSIVE single answer.
    3.  Your final output MUST be a SINGLE JSON object. Do NOT generate multiple JSON objects.
    4.  The JSON object must have the following keys:
        - "answer": (string) The single, synthesized answer.
        - "is_relevant": (boolean) True if any of the snippets contained relevant information to form an answer, otherwise false.
        - "source_page": (integer) The page number of the MOST RELEVANT snippet that contributed to the answer.
        - "source_paragraph": (integer) The paragraph number of the MOST RELEVANT snippet.
    5.  If no relevant information is found, set "is_relevant" to false.

    CONTEXT:
    ---
    {context}
    ---
    USER QUESTION: {query}

    SINGLE JSON RESPONSE:
    """


    try:
 # Request a response from Groq, enforcing JSON output format.
        res = groq_client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-emitting data extraction bot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        data = json.loads(res.choices[0].message.content.strip())

        if not data.get("is_relevant"):
            return DocumentAnswer(doc_id=doc_id)
# Create a Pydantic model object from the validated JSON data.
        return DocumentAnswer(
            doc_id=doc_id,
            extracted_answer=data.get("answer", "Could not extract answer from JSON."),
            source_page=data.get("source_page"),
            source_paragraph=data.get("source_paragraph")
        )

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[JSON Error] Doc {doc_id}: {e}")
        print("Raw model output:", res.choices[0].message.content.strip())
        return DocumentAnswer(doc_id=doc_id, extracted_answer="Error: Failed to parse model output.", source_page= None, source_paragraph=None)
    except Exception as e:
        print(f"[Groq Error] Doc {doc_id}: {e}")
        return DocumentAnswer(doc_id=doc_id, extracted_answer=f"Error processing this document: {e}")
def synthesize_themes(query: str, answers: List[DocumentAnswer]) -> List[Theme]:
# Filter out irrelevant or error-containing answers before synthesis.
    relevant_answers = [a for a in answers if "No relevant" not in a.extracted_answer and "Error" not in a.extracted_answer]
    if not relevant_answers:
        return []

    context = "\n\n".join([f"<answer doc_id='{a.doc_id}'>\n{a.extracted_answer}\n</answer>" for a in relevant_answers])
# This prompt provides a "one-shot" example of the desired JSON structure,
# which is a very effective way to guide the LLM.
    prompt = f"""
    You are a research analyst. Identify 2-4 distinct themes from the answers provided for the query: '{query}'.
    Respond with a single JSON object: {{"themes": [{{"theme_title": "string", "summary": "string", "supporting_docs": ["doc_id_1", "doc_id_2"]}}]}}.
    Ensure every theme object has all three keys.

    ANSWERS:
    ---
    {context}
    ---
    JSON RESPONSE:
    """
    try:
        res = groq_client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-emitting research analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        
 # --- Robust Parsing Logic ---
# This loop defensively checks the LLM's output before creating Pydantic models.
 # It prevents crashes if the LLM generates incomplete or malformed theme objects.
        validated_themes = []
        raw_themes = data.get("themes", [])
        for theme_data in raw_themes:
# Check if all required keys are present in the dictionary from the LLM.    
            if all(key in theme_data for key in ["theme_title", "summary", "supporting_docs"]):
                try:    
                    validated_themes.append(Theme(**theme_data))
                except Exception as pydantic_error:
                    print(f"Warning: Skipping theme due to Pydantic validation error: {pydantic_error}")
            else:
                print(f"Warning: Skipping theme due to missing fields. Data: {theme_data}")
        return validated_themes

    except Exception as e:
        print(f"ERROR in synthesize_themes: {e}")
        # Corrected error handling to match Pydantic model
        return [Theme(theme_title="API or Parsing Error", summary=f"Could not synthesize themes: {e}", supporting_docs=[])]