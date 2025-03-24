"""
Document Retrieval System

This module implements text retrieval methods using both TF-IDF and neural embeddings.
It provides functionality to rank documents based on relevance to a query.
MPS support is enabled for Apple Silicon devices.
"""

import math
import numpy as np
import torch
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModel


class DocumentRetriever:
    """A class for retrieving and ranking documents based on similarity to queries."""
    
    def __init__(self, document_collection: List[Dict[str, Any]], model_name: str = "answerdotai/ModernBERT-base", use_mps: bool = True):
        """
        Initialize the document retrieval system.
        
        Args:
            document_collection: A list of documents containing at least a 'text' field
            model_name: Transformer model name to use for neural embeddings
            use_mps: Whether to use MPS acceleration if available
        """
        self.documents = document_collection
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        self.device = self._get_device(use_mps)
        
    def _get_device(self, use_mps: bool) -> torch.device:
        """
        Determine the appropriate device for computation.
        
        Args:
            use_mps: Whether to use MPS if available
            
        Returns:
            The torch device to use
        """
        if use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    def load_neural_model(self) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load the neural embedding model and tokenizer if not already loaded.
        
        Returns:
            The model and tokenizer objects
        """
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
        
        return self.model, self.tokenizer
        
    @staticmethod
    def term_frequency(document: str, term: str) -> float:
        """
        Calculate term frequency with logarithmic scaling.
        
        Args:
            document: The document text
            term: The term to calculate frequency for
            
        Returns:
            The term frequency value
        """
        count = document.count(term)
        return 0 if count == 0 else 1 + math.log(count)
    
    @lru_cache(maxsize=None)
    def inverse_document_frequency(self, term: str) -> float:
        """
        Calculate inverse document frequency.
        
        Args:
            term: The term to calculate IDF for
            
        Returns:
            The IDF value
        """
        num_docs_with_term = sum(1 for item in self.documents if term in item["text"])
        return math.log(len(self.documents) / (1 + num_docs_with_term))
    
    def compute_tf_idf_vector(self, terms: List[str], document: str) -> List[float]:
        """
        Compute the TF-IDF vector for a document based on the given terms.
        
        Args:
            terms: The terms to include in the vector
            document: The document text
            
        Returns:
            A TF-IDF vector as a list of floats
        """
        return [self.term_frequency(document, term) * self.inverse_document_frequency(term) 
                for term in terms]
    
    @staticmethod
    def compute_cosine_similarity(vec1: Union[List[float], np.ndarray, torch.Tensor], 
                                vec2: Union[List[float], np.ndarray, torch.Tensor]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays if they're not already
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.detach().cpu().numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.detach().cpu().numpy()
            
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)

        if vec1_norm == 0 or vec2_norm == 0:
            return 0
        
        return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
    
    def get_neural_embedding(self, text: str, max_length: int = 8192) -> torch.Tensor:
        """
        Get neural embedding for a text using the loaded model.
        
        Args:
            text: The text to embed
            max_length: Maximum token length to process
            
        Returns:
            Tensor embedding for the text
        """
        model, tokenizer = self.load_neural_model()
        
        # Truncate text if needed to avoid excessive processing
        truncated_text = text[:max_length * 4]  # Rough estimate for token/char ratio
        
        with torch.no_grad():
            encoded = tokenizer(truncated_text, return_tensors="pt", truncation=True, 
                                max_length=max_length)
            # Move input tensors to the appropriate device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = model(**encoded)
            # Use the [CLS] token embedding as the text representation
            return outputs.last_hidden_state[0, 0]
    
    def rank_by_tf_idf(self, query: str, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rank documents by TF-IDF similarity to the query.
        
        Args:
            query: The search query
            top_n: Number of top results to return (None for all)
            
        Returns:
            List of ranked documents
        """
        query_terms = query.split()
        query_vec = self.compute_tf_idf_vector(query_terms, query)
        
        
        post_sort = sorted(self.documents, key=lambda x: self.compute_cosine_similarity(
                        query_vec, 
                        self.compute_tf_idf_vector(query_terms, x["text"])), reverse = True)
        
        return post_sort[:top_n]

    
    def rank_by_neural_similarity(self, docs: List[Dict[str, Any]], query: str, 
                                 top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rank documents by neural embedding similarity to the query.
        
        Args:
            docs: The documents to rank
            query: The search query
            top_n: Number of top results to return (None for all)
            
        Returns:
            List of ranked documents
        """
        query_vec = self.get_neural_embedding(query)
        
        # Create a list of (document, similarity) tuples
        ranked_docs = []
        for doc in docs:
            # Limit text length for efficiency
            doc_vec = self.get_neural_embedding(doc['text'])
            similarity = self.compute_cosine_similarity(query_vec, doc_vec)
            ranked_docs.append((doc, similarity))
        
        # Sort by similarity in descending order
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the documents, not the scores
        results = [doc for doc, _ in ranked_docs]
        
        # Limit to top_n if specified
        if top_n is not None:
            return results[:top_n]
        return results
    
    def retrieve_documents(self, query: str, tf_idf_top_n: int = 20, 
                          final_top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using a two-stage ranking process:
        1. First filter by TF-IDF
        2. Then re-rank the top results using neural embeddings
        
        Args:
            query: The search query
            tf_idf_top_n: Number of documents to retrieve with TF-IDF
            final_top_n: Number of final results to return after neural re-ranking
            
        Returns:
            List of ranked documents
        """
        # First ranking stage using TF-IDF
        top_tf_idf_results = self.rank_by_tf_idf(query, top_n=tf_idf_top_n)
        
        # Second ranking stage using neural embeddings
        final_results = self.rank_by_neural_similarity(top_tf_idf_results, query, top_n=final_top_n)
        
        return final_results


# Examples of usage
if __name__ == "__main__":
    # This code will only run when the script is executed directly, not when imported
    from datasets import load_dataset
    
    # Load datasets
    wiki = load_dataset("nuprl/engineering-llm-systems", 
                       name="wikipedia-northeastern-university", 
                       split="test")
    
    questions = load_dataset("nuprl/engineering-llm-systems", 
                            name="obscure_questions", 
                            split="tiny")
    
    # Print device information
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create document retriever with MPS enabled
    retriever = DocumentRetriever(wiki, use_mps=True)
    print(f"Using device: {retriever.device}")
    
    # Example 1: TF-IDF ranking
    cs_profs = retriever.rank_by_tf_idf("Northeastern University computer science professor", top_n=20)
    print("Top 20 results by TF-IDF:")
    for item in cs_profs:
        print(f"{item['title']} {item['url']}")
    
    # Example 2: Neural ranking
    neural_ranked = retriever.rank_by_neural_similarity(
        cs_profs, "Northeastern University computer science professor")
    print("\nTop results by neural similarity:")
    for item in neural_ranked[:5]:
        print(f"{item['title']} {item['url']}")
    
    # Example 3: Combined retrieval
    print("\nCombined retrieval results:")
    test_results = retriever.retrieve_documents("Who was the best pitcher in the American League")
    for item in test_results:
        print(f"{item['title']} {item['url']}")