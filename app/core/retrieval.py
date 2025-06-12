import chromadb
from typing import List, Dict, Any
from .utils import DocumentSnippet

class VectorStore:
    def __init__(self, collection_name: str = "document_research"):
        self.client = chromadb.Client() # In-memory client
        # To persist to disk, use: self.client = chromadb.PersistentClient(path="/path/to/db")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, snippets: List[DocumentSnippet]):
        """Adds document snippets to the vector store."""
        if not snippets:
            return

        ids = [f"{s.doc_id}_p{s.page}_para{s.paragraph}" for s in snippets]
        documents = [s.content for s in snippets]
        metadatas = [s.dict(exclude={'content'}) for s in snippets]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(snippets)} snippets to the vector store.")

    def search_in_document(self, query: str, doc_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Searches for relevant snippets within a specific document."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"doc_id": doc_id}
        )
        
        # Unpack results
        if not results or not results['documents']:
            return []

        retrieved_snippets = []
        for i in range(len(results['documents'][0])):
            retrieved_snippets.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        return retrieved_snippets

    def get_all_doc_ids(self) -> List[str]:
        """Returns a list of all unique document IDs in the store."""
        all_metadatas = self.collection.get(include=["metadatas"])['metadatas']
        if not all_metadatas:
            return []
        doc_ids = {meta['doc_id'] for meta in all_metadatas}
        return list(doc_ids)
    
    def clear_database(self):
        """Deletes the collection, clearing all data."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        print("Vector database cleared.")
