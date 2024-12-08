# Specifiche Progetto: Analisi Documenti PDF ed Estrazione Frasi con Parole Chiave

## Obiettivo

Sviluppare un'applicazione Python per analizzare documenti PDF e identificare frasi contenenti specifiche parole chiave, fornendo contesto e rilevando date nelle vicinanze.

## Funzionalità Principali

### Input

1. Cartella contenente i file PDF da analizzare
2. File Excel (.xlsx) contenente la lista delle parole chiave
3. Parametri di configurazione forniti da linea di comando:
    - Numero di frasi da estrarre prima della frase target
    - Numero di frasi da estrarre dopo la frase target
    - Range di ricerca per le date (numero di frasi in cui cercare)

### Processo

1. Lettura ed elaborazione di tutti i PDF nella cartella specificata
2. Estrazione del testo da ogni PDF
3. Segmentazione del testo in frasi
4. Identificazione delle frasi contenenti le parole chiave
5. Estrazione del contesto (frasi prima e dopo) secondo i parametri forniti
6. Ricerca di date nel range specificato
7. Generazione degli output in formato Excel

### Output

1. Un file Excel separato per ogni PDF analizzato:
    - Nome file: basato sul titolo del PDF analizzato
    - Contenuto: risultati dell'analisi per quel specifico PDF
2. Un file Excel complessivo contenente:
    - Risultati di tutti i PDF analizzati
    - Colonna specifica per identificare il PDF di origine di ogni frase

## Interfaccia Utente

### Linea di Comando

L'applicazione sarà eseguita da terminale con i seguenti parametri (in italiano):

```bash
python main.py \\
  --cartella_input ./pdf_docs \\
  --file_parole_chiave ./keywords.xlsx \\
  --output risultati.xlsx \\
  --frasi_prima 2 \\
  --frasi_dopo 2 \\
  --range_date 3

```

### Struttura Output Excel

Ogni file Excel (sia individuali che complessivo) conterrà:

- Nome del documento PDF di origine
- Parola chiave trovata
- Frase contenente la parola chiave
- Frasi di contesto precedenti
- Frasi di contesto successive
- Data identificata (se presente)

## Architettura Tecnica

### Componenti Core

1. **Gestione PDF**
    - Lettura file PDF (PyPDF2 o pdfminer.six)
    - Estrazione testo
    - Gestione encoding e caratteri speciali
2. **Elaborazione Testo**
    - Preprocessing e pulizia
    - Segmentazione in frasi (NLTK o spaCy)
    - Normalizzazione
3. **Analisi Contenuto**
    - Ricerca parole chiave
    - Estrazione contesto
    - Identificazione date (dateparser)
4. **Gestione Output**
    - Creazione DataFrame Pandas
    - Generazione file Excel individuali
    - Generazione file Excel complessivo

### Funzionalità Opzionali

- Integrazione RAG per ricerca semantica (se configurato internamente)
- Logging per debug e monitoraggio
- Possibilità future di espansione per:
    - Analisi sentiment
    - Supporto multilingua
    - Classificazione automatica
    - Miglioramento comprensione semantica

## Integrazione AI/ML

### Componenti RAG (Retrieval Augmented Generation)

1. **Text Embedding**
    - Modello: Sentence-BERT (specificamente il modello `all-MiniLM-L6-v2`)
    - Funzionalità:
        - Generazione embedding per le frasi estratte dai PDF
        - Generazione embedding per le parole chiave
        - Dimensione embedding: 384 dimensioni
    - Vantaggi:
        - Velocità di elaborazione
        - Ottimo bilanciamento tra performance e risorse
        - Supporto multilingua
2. **Vector Store**
    - FAISS (Facebook AI Similarity Search)
        - Indice: IndexFlatL2 per ricerca euclidea
        - Configurazione: dimensione=384, nlist=100
    - Alternative:
        - Chroma (in-memory per dataset piccoli)
        - Qdrant (per deployment in produzione)
    - Funzionalità:
        - Memorizzazione efficiente degli embedding
        - Ricerca veloce dei nearest neighbors
        - Possibilità di persistenza su disco
3. **LLM Integration**
    - Modello Primario: Mistral-7B-Instruct
        - Quantizzazione: GGUF 4-bit
        - Contesto: 8k tokens
        - Vantaggio: ottimo bilanciamento performance/risorse
    - Alternative:
        - Phi-2 (per tasks più leggeri)
        - Mixtral-8x7B (per maggiore accuratezza)
    - Compiti:
        - Validazione semantica delle corrispondenze
        - Analisi contestuale delle frasi
        - Identificazione relazioni implicite

### Pipeline di Elaborazione Semantica

1. **Preprocessing Avanzato**
    
    ```python
    def semantic_preprocessing(text):
        # Normalizzazione del testo
        text = normalize_text(text)
    
        # Segmentazione in frasi con spaCy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
    
        # Generazione embedding per ogni frase
        embeddings = model.encode(sentences,
                                batch_size=32,
                                show_progress_bar=True)
    
        return sentences, embeddings
    
    ```
    
2. **Ricerca Semantica**
    
    ```python
    def semantic_search(query, embeddings, sentences, threshold=0.7):
        # Encoding della query
        query_embedding = model.encode(query)
    
        # Ricerca dei nearest neighbors
        D, I = index.search(query_embedding, k=5)
    
        # Filtro per similarità
        results = [
            (sentences[i], score)
            for i, score in zip(I[0], D[0])
            if score < threshold
        ]
    
        return results
    
    ```
    
3. **Validazione LLM**
    
    ```python
    def llm_validation(context, matches):
        prompt = f"""
        Contesto: {context}
        Frasi trovate: {matches}
    
        Valuta la rilevanza semantica delle frasi trovate rispetto al contesto.
        Restituisci solo le frasi veramente pertinenti.
        """
    
        response = llm(prompt,
                      max_tokens=500,
                      temperature=0.3)
    
        return parse_llm_response(response)
    
    ```
    

### Configurazione e Parametri

1. **Embedding Configuration**
    
    ```python
    EMBEDDING_CONFIG = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_seq_length": 256,
        "normalize_embeddings": True
    }
    
    ```
    
2. **Vector Store Settings**
    
    ```python
    VECTOR_STORE_CONFIG = {
        "index_type": "IndexFlatL2",
        "dimension": 384,
        "nlist": 100,
        "metric_type": "L2"
    }
    
    ```
    
3. **LLM Parameters**
    
    ```python
    LLM_CONFIG = {
        "model_path": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "context_length": 8192,
        "temperature": 0.3,
        "top_p": 0.95,
        "repeat_penalty": 1.1
    }
    
    ```
    

### Ottimizzazioni e Performance

1. **Batch Processing**
    - Elaborazione embedding in batch (32-64 frasi)
    - Caching degli embedding su disco
    - Parallelizzazione della generazione embedding
2. **Memory Management**
    - Streaming dei PDF per file grandi
    - Garbage collection proattivo
    - Gestione efficiente del contesto LLM
3. **Caching Strategy**
    
    ```python
    CACHE_CONFIG = {
        "embedding_cache": {
            "type": "sqlite",
            "path": "./cache/embeddings.db",
            "max_size": "1GB"
        },
        "vector_store_cache": {
            "type": "mmap",
            "path": "./cache/vectors/"
        }
    }
    
    ```
    

### Metriche e Valutazione

1. **Qualità Semantica**
    - Precision@K per risultati semantici
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
2. **Performance**
    - Latenza media per query
    - Throughput documenti/minuto
    - Utilizzo memoria/CPU
3. **Monitoraggio**
    
    ```python
    METRICS_CONFIG = {
        "semantic_metrics": ["precision", "recall", "f1"],
        "performance_metrics": ["latency", "throughput"],
        "resource_metrics": ["memory", "cpu_usage"]
    }
    
    ```
    

## Note Tecniche

### Librerie Principali

- PyPDF2 o pdfminer.six per gestione PDF
- Pandas per gestione dati e output Excel
- NLTK o spaCy per elaborazione linguistica
- dateparser per identificazione date
- LangChain (opzionale) per funzionalità RAG

### Considerazioni Implementative

- Gestione efficiente della memoria per file PDF grandi
- Normalizzazione testo e punteggiatura
- Mantenimento dell'ordine delle frasi per estrazione contesto
- Gestione errori e casi limite
- Performance ottimizzate per grandi volumi di documenti

## Limitazioni e Vincoli

- Nessuna interfaccia grafica
- Configurazioni avanzate (RAG, embedding, ecc.) predefinite nel codice
- No configurazioni esterne (JSON/YAML)