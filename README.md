# Documentazione Tecnica - PDF Analyzer
## Versione 1.0

### 1. Panoramica del Sistema
Il sistema PDF Analyzer è un'applicazione Python progettata per analizzare documenti PDF ed estrarre frasi contenenti specifiche parole chiave, con funzionalità di analisi semantica e contestuale.

### 2. Specifiche Tecniche e Requisiti
#### 2.1 Requisiti di Sistema
- Python 3.7 o superiore
- RAM: 8GB minimo consigliato 
- Storage: Spazio sufficiente per i file di input/output
- OS: Windows/Linux/MacOS

#### 2.2 Dipendenze Software
- pandas
- PyPDF2
- sentence-transformers
- torch
- langchain
- tqdm

### 3. Funzionalità Implementate
#### 3.1 Core Features
- Estrazione testo da PDF
- Ricerca di keyword (esatta e semantica)
- Analisi contestuale (n frasi prima/dopo)
- Riconoscimento date nel contesto
- Generazione report Excel

#### 3.2 Caratteristiche Avanzate
- Analisi semantica tramite modelli transformer
- Deduplicazione risultati
- Sistema di scoring per rilevanza
- Logging dettagliato
- Gestione errori robusta

### 4. Struttura Input/Output
#### 4.1 File di Input
- PDF da analizzare in cartella `data/input`
- Keywords in file Excel `data/words/keywords.xlsx`
- Configurazioni in `config.py`

#### 4.2 File di Output
- Report Excel per singolo PDF
- Report Excel complessivo
- Log dettagliato delle operazioni

### 5. Guida all'Installazione e Uso

```bash
# 1. Creare ambiente virtuale
python -m venv venv_py310

# 2. Attivare ambiente virtuale
# Windows:
venv_py310\Scripts\activate
# Linux/Mac:
source venv_py310/bin/activate

# 3. Installare dipendenze
pip install -r requirements.txt

# 4. Preparare struttura cartelle
mkdir -p data/input data/output data/words

# 5. Inserire files
# - PDF in data/input/
# - Keywords Excel in data/words/

# 6. Eseguire applicazione
python main.py
```

### 6. Analisi dei Risultati e Limitazioni Attuali

#### 6.1 Qualità Output
- Buona capacità di estrazione keyword esatte
- Discreta analisi semantica
- Presenza di alcuni falsi positivi
- Possibili duplicazioni nei risultati

#### 6.2 Limitazioni Note
- Sensibilità alla qualità del PDF input
- Estrazione testo da indici e headers
- Scoring semantico da raffinare
- Performance su PDF molto grandi

### 7. Raccomandazioni per Miglioramenti

#### 7.1 Input PDF
- Evitare indici se possibile
- Usare PDF con testo ben strutturato
- Evitare tabelle complesse
- Minimizzare headers/footers ripetitivi

#### 7.2 Keywords
- Usare keyword specifiche e non ambigue
- Evitare keyword troppo generiche
- Considerare varianti linguistiche
- Strutturare per tematiche

#### 7.3 Sviluppi Futuri Suggeriti
- Miglior preprocessing del testo
- Raffinamento scoring semantico
- Deduplicazione più sofisticata
- Interfaccia utente grafica
- Supporto multilingua avanzato

### 8. Note Tecniche Aggiuntive

#### 8.1 Performance
- Tempi elaborazione: ~30s per PDF medio
- Memoria: ~2GB durante elaborazione
- CPU/GPU: ottimizzabile con GPU

#### 8.2 Sicurezza
- Elaborazione locale dei dati
- No comunicazione rete (post setup)
- Log rotazione automatica

