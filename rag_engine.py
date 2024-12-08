import torch
from typing import List, Tuple, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from dataclasses import dataclass
from config import config, RAGConfig
import logging
from contextlib import contextmanager
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class SemanticMatch:
    """Rappresenta una corrispondenza semantica trovata"""
    text: str
    score: float
    context: Optional[str] = None
    metadata: Optional[dict] = None

class RAGEngine:
    """Motore RAG avanzato per l'analisi semantica"""
    
    def __init__(self, rag_config: RAGConfig = None):
        self.config = rag_config or config.rag
        logger.info("Inizializzazione RAG Engine...")
        
        try:
            # Determina il dispositivo ottimale
            self.device = self._get_optimal_device()
            logger.info(f"Usando dispositivo: {self.device}")
            
            # Inizializza il modello di embedding
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs={'device': self.device}
            )
            
            # Cache per gli embedding
            self.embedding_cache = {}
            
            logger.info("RAG Engine inizializzato con successo")
            
        except Exception as e:
            logger.error(f"Errore nell'inizializzazione RAG Engine: {e}")
            raise

    def _get_optimal_device(self) -> str:
        """Determina il dispositivo ottimale per il calcolo"""
        if torch.cuda.is_available():
            logger.info("CUDA disponibile - usando GPU")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS disponibile - usando Apple Silicon")
            return 'mps'
        else:
            logger.info("Nessun acceleratore disponibile - usando CPU")
            return 'cpu'

    def _calculate_embedding(self, text: str) -> np.ndarray:
        """Calcola l'embedding di un testo con caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embeddings.embed_query(text)
        self.embedding_cache[text] = embedding
        return embedding

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcola la similarità semantica tra due testi"""
        try:
            emb1 = self._calculate_embedding(text1)
            emb2 = self._calculate_embedding(text2)
            
            # Calcola similarità del coseno
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Errore nel calcolo della similarità: {e}")
            return 0.0

    def find_semantic_matches(self, 
                            query: str, 
                            texts: List[str], 
                            threshold: float = None,
                            top_k: int = None) -> List[SemanticMatch]:
        """Trova i match semantici più rilevanti"""
        threshold = threshold or self.config.similarity_threshold
        top_k = top_k or self.config.retrieval_k
        
        try:
            # Crea un indice FAISS temporaneo
            vectorstore = FAISS.from_texts(texts, self.embeddings)
            
            # Cerca i documenti più simili
            results = vectorstore.similarity_search_with_score(query, k=top_k)
            
            matches = []
            for doc, score in results:
                if score > threshold:
                    match = SemanticMatch(
                        text=doc.page_content,
                        score=float(score),
                        metadata=doc.metadata
                    )
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"Errore nella ricerca semantica: {e}")
            return []

    def analyze_semantic_context(self, 
                               target_text: str, 
                               context_texts: List[str], 
                               window_size: int = 3) -> List[SemanticMatch]:
        """Analizza il contesto semantico di un testo target"""
        try:
            # Prepara finestre di contesto
            context_windows = []
            for i in range(len(context_texts) - window_size + 1):
                window = context_texts[i:i + window_size]
                context_windows.append(' '.join(window))
            
            # Trova i contesti più rilevanti
            matches = self.find_semantic_matches(
                target_text,
                context_windows,
                threshold=self.config.reranking_threshold
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Errore nell'analisi del contesto: {e}")
            return []

    def batch_process_texts(self, 
                          texts: List[str], 
                          keywords: List[str], 
                          batch_size: int = 32) -> List[Tuple[str, List[SemanticMatch]]]:
        """Processa un batch di testi per trovare corrispondenze semantiche"""
        results = []
        
        try:
            # Processa in batch per efficienza
            for i in tqdm(range(0, len(texts), batch_size), desc="Analisi semantica batch"):
                batch = texts[i:i + batch_size]
                batch_results = []
                
                for text in batch:
                    matches = self.find_semantic_matches(text, keywords)
                    if matches:
                        batch_results.append((text, matches))
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Errore nel processing batch: {e}")
            return []

    def cleanup(self):
        """Pulisce la cache e libera risorse"""
        try:
            self.embedding_cache.clear()
            torch.cuda.empty_cache()
            logger.info("Pulizia cache completata")
        except Exception as e:
            logger.error(f"Errore nella pulizia cache: {e}")

@contextmanager
def rag_engine_context():
    """Context manager per gestire il ciclo di vita del RAG Engine"""
    engine = None
    try:
        engine = RAGEngine()
        yield engine
    finally:
        if engine:
            engine.cleanup()