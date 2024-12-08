from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """Configurazione per il sistema RAG"""
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7
    max_context_length: int = 1000
    retrieval_k: int = 3  # Numero di documenti simili da recuperare
    reranking_threshold: float = 0.6

@dataclass
class ProcessingConfig:
    """Configurazione per il processing del testo"""
    min_sentence_length: int = 10
    max_sentence_length: int = 1000
    default_context_size: int = 2
    default_date_window: int = 3
    valid_languages: list = None  # None significa tutte le lingue

    def __post_init__(self):
        if self.valid_languages is None:
            self.valid_languages = ['it', 'en']

@dataclass
class TimeoutConfig:
    """Configurazione per i timeout"""
    pdf_extraction_timeout: int = 30  # secondi
    semantic_analysis_timeout: int = 5
    rag_search_timeout: int = 10
    total_pdf_timeout: int = 300  # 5 minuti per PDF

@dataclass
class RetryConfig:
    """Configurazione per i retry"""
    max_attempts: int = 3
    delay: int = 1
    exponential_backoff: bool = True

class AppConfig:
    """Configurazione globale dell'applicazione"""
    
    def __init__(self):
        self.rag = RAGConfig()
        self.processing = ProcessingConfig()
        self.timeout = TimeoutConfig()
        self.retry = RetryConfig()
        
        # Dizionario di sinonimi comuni in italiano
        self.common_synonyms = {
            'ambiente': ['environmental', 'ambientale', 'environmentally'],
            'sostenibile': ['sostenibilità', 'sustainable', 'sustainability'],
            'innovazione': ['innovativo', 'innovative', 'innovation'],
            'ricerca': ['research', 'ricercare', 'researching'],
            'sviluppo': ['development', 'developing', 'develop'],
        }
        
        # Configurazione formati date validi
        self.date_formats = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{2,4}',
            r'\d{1,2}\s+(?:gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\s+\d{2,4}',
            # Formati internazionali
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{2}/\d{2}/\d{4}',  # Common EU/US format
        ]
        
        # Mapping mesi per normalizzazione date
        self.month_mapping = {
            'gennaio': '01', 'gen': '01',
            'febbraio': '02', 'feb': '02',
            'marzo': '03', 'mar': '03',
            'aprile': '04', 'apr': '04',
            'maggio': '05', 'mag': '05',
            'giugno': '06', 'giu': '06',
            'luglio': '07', 'lug': '07',
            'agosto': '08', 'ago': '08',
            'settembre': '09', 'set': '09',
            'ottobre': '10', 'ott': '10',
            'novembre': '11', 'nov': '11',
            'dicembre': '12', 'dic': '12'
        }

    def to_dict(self) -> Dict[str, Any]:
        """Converte la configurazione in dizionario"""
        return {
            'rag': self.rag.__dict__,
            'processing': self.processing.__dict__,
            'timeout': self.timeout.__dict__,
            'retry': self.retry.__dict__,
            'common_synonyms': self.common_synonyms,
            'date_formats': self.date_formats,
            'month_mapping': self.month_mapping
        }

# Istanza globale della configurazione
config = AppConfig()