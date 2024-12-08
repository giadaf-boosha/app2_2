import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import logging
import unicodedata
from tqdm import tqdm
import time
from functools import wraps
import signal
from contextlib import contextmanager
import traceback
from typing import Callable
import numpy as np
import torch
from config import config
from rag_engine import RAGEngine, SemanticMatch, rag_engine_context

# Configurazione logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutError(f"L'operazione ha superato il timeout di {seconds} secondi")

    # Imposta il gestore del segnale
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def retry(max_attempts: int = 3, delay: int = 1):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Tentativo {attempt + 1}/{max_attempts} fallito: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            logger.error(f"Tutti i tentativi falliti: {str(last_exception)}")
            raise last_exception
        return wrapper
    return decorator

@dataclass
class MatchedPhrase:
    document: str
    keyword: str
    phrase: str
    context_before: List[str]
    context_after: List[str]
    semantic_similarity: float = 0.0
    semantic_context: List[str] = None
    semantic_context_scores: List[float] = None
    date: Optional[str] = None

    def __post_init__(self):
        if self.semantic_context is None:
            self.semantic_context = []
        if self.semantic_context_scores is None:
            self.semantic_context_scores = []

class SemanticProcessor:
    """Gestisce l'analisi semantica del testo utilizzando RAGEngine"""
    
    def __init__(self):
        logger.info("Inizializzazione del processore semantico...")
        try:
            self.rag_engine = RAGEngine(config.rag)
            logger.info("Processore semantico inizializzato con successo")
        except Exception as e:
            logger.error(f"Errore nel caricamento del processore semantico: {e}")
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcola la similarità semantica tra due testi"""
        try:
            with timeout(config.timeout.semantic_analysis_timeout):
                return self.rag_engine.calculate_similarity(text1, text2)
        except TimeoutError:
            logger.warning("Timeout nel calcolo della similarità semantica")
            return 0.0
        except Exception as e:
            logger.error(f"Errore nel calcolo della similarità: {e}")
            return 0.0

    def find_semantic_matches(self, text: str, keywords: List[str]) -> List[Tuple[str, float]]:
        """Trova le parole chiave semanticamente simili nel testo"""
        try:
            with timeout(config.timeout.rag_search_timeout):
                matches = self.rag_engine.find_semantic_matches(
                    text,
                    keywords,
                    threshold=config.rag.similarity_threshold
                )
                return [(m.text, m.score) for m in matches]
        except TimeoutError:
            logger.warning("Timeout nella ricerca semantica")
            return []
        except Exception as e:
            logger.error(f"Errore nella ricerca semantica: {e}")
            return []

    def analyze_semantic_context(self, text: str, context_sentences: List[str]) -> List[Tuple[str, float]]:
        """Analizza il contesto semantico di una frase"""
        try:
            with timeout(config.timeout.semantic_analysis_timeout):
                context_matches = self.rag_engine.analyze_semantic_context(
                    text,
                    context_sentences,
                    window_size=config.processing.default_context_size
                )
                return [(m.text, m.score) for m in context_matches]
        except TimeoutError:
            logger.warning("Timeout nell'analisi del contesto semantico")
            return []
        except Exception as e:
            logger.error(f"Errore nell'analisi del contesto semantico: {e}")
            return []

    def cleanup(self):
        """Pulisce le risorse del RAG Engine"""
        try:
            self.rag_engine.cleanup()
        except Exception as e:
            logger.error(f"Errore nella pulizia delle risorse: {e}")

class TextPreprocessor:
    """Classe per la pulizia e normalizzazione del testo"""
    
    @staticmethod
    @retry(max_attempts=3)
    def normalize_text(text: str) -> str:
        """Normalizza il testo per il matching"""
        try:
            # Converti in minuscolo
            text = text.lower()
            # Normalizza caratteri Unicode
            text = unicodedata.normalize('NFKD', text)
            text = ''.join(c for c in text if not unicodedata.combining(c))
            # Normalizza spazi
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Errore nella normalizzazione del testo: {e}")
            raise

class PDFAnalyzer:
    def __init__(self, context_size: int = 2, date_window: int = 3):
        self.context_size = context_size
        self.date_window = date_window
        self.text_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        self.preprocessor = TextPreprocessor()
        self.semantic_processor = SemanticProcessor()
        logger.info("PDFAnalyzer inizializzato con successo")

    @retry(max_attempts=3)
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Estrae il testo da un PDF con gestione errori"""
        logger.info(f"Inizio estrazione testo da: {pdf_path}")
        try:
            with timeout(30):  # timeout di 30 secondi per PDF
                reader = PdfReader(pdf_path)
                text = ""
                for i, page in enumerate(tqdm(reader.pages, desc="Lettura pagine")):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.error(f"Errore nell'estrazione della pagina {i}: {e}")
                return self.preprocessor.normalize_text(text)
        except TimeoutError:
            logger.error(f"Timeout nell'estrazione del testo da {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Errore nell'estrazione del testo: {e}\n{traceback.format_exc()}")
            raise

    def split_into_sentences(self, text: str) -> List[str]:
        """Divide il testo in frasi con logging"""
        logger.info("Inizio segmentazione in frasi")
        try:
            sentences = []
            for chunk in self.text_splitter.split_text(text):
                if chunk.strip():
                    sentences.append(chunk.strip() + ".")
            logger.info(f"Segmentazione completata: {len(sentences)} frasi trovate")
            return sentences
        except Exception as e:
            logger.error(f"Errore nella segmentazione: {e}\n{traceback.format_exc()}")
            raise

    @retry(max_attempts=2)
    def find_dates(self, text: str) -> Optional[str]:
        """Cerca date nel testo con retry"""
        try:
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{2,4}',
                r'\d{1,2}\s+(?:gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)\s+\d{2,4}'
            ]

            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
            return None
        except Exception as e:
            logger.error(f"Errore nella ricerca date: {e}")
            return None

    def process_pdf_documents(self, pdf_folder: str, keywords_file: str, output_folder: str) -> None:
        """Elabora i documenti PDF con gestione errori completa"""
        start_time = time.time()
        logger.info("Inizio elaborazione documenti PDF")

        try:
            # Carica keywords
            logger.info("Lettura file keywords...")
            keywords_df = pd.read_excel(keywords_file)
            keywords = keywords_df[keywords_df.columns[0]].astype(str).tolist()
            keywords = [k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()]
            logger.info(f"Caricate {len(keywords)} keywords")

            all_results = []
            pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
            
            for pdf_file in tqdm(pdf_files, desc="Elaborazione PDF"):
                pdf_start = time.time()
                logger.info(f"Inizio analisi: {pdf_file}")
                
                try:
                    with timeout(300):  # timeout di 5 minuti per PDF
                        pdf_path = os.path.join(pdf_folder, pdf_file)
                        text = self.extract_text_from_pdf(pdf_path)
                        sentences = self.split_into_sentences(text)
                        
                        pdf_results = []
                        logger.info("Inizio ricerca keywords e analisi semantica...")
                        
                        for idx, sentence in enumerate(tqdm(sentences, desc="Analisi frasi")):
                            # Ricerca esatta
                            sentence_lower = sentence.lower()
                            for keyword in keywords:
                                if keyword in sentence_lower:
                                    semantic_score = self.semantic_processor.calculate_similarity(sentence, keyword)
                                    match = self._create_match(pdf_file, keyword, sentence, sentences, idx, semantic_score)
                                    pdf_results.append(match)
                            
                            # Ricerca semantica
                            semantic_matches = self.semantic_processor.find_semantic_matches(sentence, keywords)
                            for keyword, similarity in semantic_matches:
                                if similarity > 0.7 and keyword not in sentence_lower:  # evita duplicati
                                    match = self._create_match(pdf_file, keyword, sentence, sentences, idx, similarity)
                                    pdf_results.append(match)
                        
                        if pdf_results:
                            logger.info(f"Trovate {len(pdf_results)} corrispondenze in {pdf_file}")
                            output_path = os.path.join(output_folder, f"report_{pdf_file}.xlsx")
                            self._create_excel_report(pdf_results, output_path)
                            all_results.extend(pdf_results)
                        
                        pdf_time = time.time() - pdf_start
                        logger.info(f"PDF completato in {pdf_time:.1f} secondi")
                        
                except TimeoutError:
                    logger.error(f"Timeout nell'elaborazione di {pdf_file}")
                except Exception as e:
                    logger.error(f"Errore nell'elaborazione di {pdf_file}: {e}\n{traceback.format_exc()}")
                    continue

            if all_results:
                output_path = os.path.join(output_folder, "report_completo.xlsx")
                self._create_excel_report(all_results, output_path)
                logger.info(f"Creato report complessivo con {len(all_results)} risultati")
            
            total_time = time.time() - start_time
            logger.info(f"Elaborazione completata in {total_time:.1f} secondi")

        except Exception as e:
            logger.error(f"Errore critico nell'elaborazione: {e}\n{traceback.format_exc()}")
            raise

    def _create_match(self, pdf_file: str, keyword: str, sentence: str, sentences: List[str], 
                     idx: int, semantic_score: float) -> MatchedPhrase:
        """Crea un oggetto MatchedPhrase con analisi del contesto semantico"""
        try:
            # Ottieni il contesto standard
            context_before = sentences[max(0, idx-self.context_size):idx]
            context_after = sentences[idx+1:idx+1+self.context_size]
            
            # Ottieni il contesto semantico
            context_window = sentences[max(0, idx-config.processing.default_context_size):
                                    min(len(sentences), idx+config.processing.default_context_size+1)]
            semantic_context = []
            semantic_context_scores = []
            
            # Analizza il contesto semantico
            semantic_matches = self.semantic_processor.analyze_semantic_context(sentence, context_window)
            if semantic_matches:
                semantic_context = [match[0] for match in semantic_matches]
                semantic_context_scores = [match[1] for match in semantic_matches]
            
            # Cerca date nel contesto
            date_context = sentences[max(0, idx-self.date_window):idx+self.date_window+1]
            date = self.find_dates(' '.join(date_context))
            
            return MatchedPhrase(
                document=pdf_file,
                keyword=keyword,
                phrase=sentence,
                context_before=context_before,
                context_after=context_after,
                semantic_similarity=semantic_score,
                semantic_context=semantic_context,
                semantic_context_scores=semantic_context_scores,
                date=date
            )
        except Exception as e:
            logger.error(f"Errore nella creazione del match: {e}")
            raise

    @retry(max_attempts=3)
    def _create_excel_report(self, results: List[MatchedPhrase], output_path: str) -> None:
        """Crea il report Excel con informazioni semantiche aggiuntive"""
        try:
            data = []
            for result in results:
                row = {
                    'Documento': result.document,
                    'Parola Chiave': result.keyword,
                    'Frase': result.phrase,
                    'Contesto Prima': ' '.join(result.context_before),
                    'Contesto Dopo': ' '.join(result.context_after),
                    'Similarità Semantica': f"{result.semantic_similarity:.2f}",
                    'Data': result.date if result.date else ''
                }

                # Aggiungi informazioni sul contesto semantico
                if result.semantic_context and result.semantic_context_scores:
                    for i, (context, score) in enumerate(zip(result.semantic_context, result.semantic_context_scores)):
                        row[f'Contesto Semantico {i+1}'] = context
                        row[f'Score Semantico {i+1}'] = f"{score:.2f}"

                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Ordina le colonne per una migliore leggibilità
            column_order = ['Documento', 'Parola Chiave', 'Frase', 'Contesto Prima', 'Contesto Dopo', 
                          'Similarità Semantica', 'Data']
            semantic_columns = [col for col in df.columns if col.startswith(('Contesto Semantico', 'Score Semantico'))]
            column_order.extend(sorted(semantic_columns))
            
            df = df[column_order]
            df.to_excel(output_path, index=False)
            logger.info(f"Report salvato: {output_path}")
        except Exception as e:
            logger.error(f"Errore nella creazione del report Excel: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Analizzatore PDF - Funzionalità 1')
    parser.add_argument('--input_folder', required=True,
                      help='Cartella contenente i file PDF da analizzare')
    parser.add_argument('--keywords_file', required=True,
                      help='File Excel contenente le parole chiave')
    parser.add_argument('--output_folder', required=True,
                      help='Cartella dove salvare i report Excel')
    parser.add_argument('--context_size', type=int, default=2,
                      help='Numero di frasi di contesto da estrarre prima e dopo')
    parser.add_argument('--date_window', type=int, default=3,
                      help='Numero di frasi in cui cercare le date')

    args = parser.parse_args()

    logger.info("\n=== Avvio Analizzatore PDF ===")
    logger.info(f"Cartella input: {args.input_folder}")
    logger.info(f"File keywords: {args.keywords_file}")
    logger.info(f"Cartella output: {args.output_folder}")
    logger.info(f"Contesto: {args.context_size} frasi")
    logger.info(f"Finestra date: {args.date_window} frasi")
    logger.info("============================\n")

    try:
        os.makedirs(args.output_folder, exist_ok=True)
        if not os.path.exists(args.input_folder):
            raise FileNotFoundError(f"La cartella {args.input_folder} non esiste")
        if not os.path.exists(args.keywords_file):
            raise FileNotFoundError(f"Il file {args.keywords_file} non esiste")

        analyzer = PDFAnalyzer(
            context_size=args.context_size,
            date_window=args.date_window
        )
        
        analyzer.process_pdf_documents(
            args.input_folder,
            args.keywords_file,
            args.output_folder
        )

    except Exception as e:
        logger.error(f"Errore critico nell'esecuzione: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nOperazione interrotta dall'utente")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Errore fatale: {e}\n{traceback.format_exc()}")
        sys.exit(1)