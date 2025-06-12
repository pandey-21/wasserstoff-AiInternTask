import chromadb
from typing import List, Dict, Any
from .utils import DocumentSnippet

# It's good practice to import the embedding function directly,
# to make the code more explicit and stable.
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, collection_name: str = "document_research"):
        """
        Initializes the VectorStore.
        It stores the collection_name to be used by other methods,
        which is crucial for session isolation.
        """
        self.client = chromadb.Client()
        self.collection_name = collection_name # Store the unique collection name

        # Define the embedding function to be used.
        # This ensures consistency and avoids conflicts.
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create the collection with the specific name and embedding function.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_func
        )

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
        print(f"Added {len(snippets)} snippets to collection '{self.collection_name}'.")

    def search_in_document(self, query: str, doc_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Searches for relevant snippets within a specific document."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"doc_id": doc_id}
        )
        
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
        # This gets IDs from the *current* collection for this session.
        if self.collection.count() == 0:
            return []
            
        all_metadatas = self.collection.get(include=["metadatas"])['metadatas']
        if not all_metadatas:
            return []
        return list({meta['doc_id'] for meta in all_metadatas})
    
    def clear_database(self):
        """
        Deletes the collection specific to this user's session.
        This uses the collection_name stored during initialization.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' cleared.")
            # Re-create the collection immediately so the app can continue working
            # without requiring a full rerun or re-initialization.
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_func
            )
        except Exception as e:
            # This can happen if the collection doesn't exist, which is not a critical error.
            print(f"Could not clear collection '{self.collection_name}' (it may not have existed): {e}")
