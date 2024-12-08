# Istruzioni di Installazione ed Esecuzione - PDF Analyzer

## Prerequisiti
- Python 3.7+ installato
- Git installato (opzionale)
- Pip installato
- Accesso a Internet per l'installazione iniziale

## Setup per Windows

```batch
# 1. Creare cartella progetto e spostarsi dentro
mkdir pdf_analyzer
cd pdf_analyzer

# 2. Creare ambiente virtuale Python 3.10
python -m venv venv_py310

# 3. Attivare ambiente virtuale
venv_py310\Scripts\activate

# 4. Installare dipendenze principali
pip install pandas PyPDF2 sentence-transformers torch langchain tqdm

# 5. Creare struttura cartelle
mkdir data
cd data
mkdir input output words
cd ..

# 6. Copiare i file necessari
# - Copiare i file .py nella cartella principale
# - Copiare i PDF da analizzare in data/input/
# - Copiare keywords.xlsx in data/words/

# 7. Eseguire il programma
python main.py
```

## Setup per Mac/Linux

```bash
# 1. Creare cartella progetto e spostarsi dentro
mkdir pdf_analyzer
cd pdf_analyzer

# 2. Creare ambiente virtuale Python 3.10
python3 -m venv venv_py310

# 3. Attivare ambiente virtuale
source venv_py310/bin/activate

# 4. Installare dipendenze principali
pip install pandas PyPDF2 sentence-transformers torch langchain tqdm

# 5. Creare struttura cartelle
mkdir -p data/input data/output data/words

# 6. Copiare i file necessari
# - Copiare i file .py nella cartella principale
# - Copiare i PDF da analizzare in data/input/
# - Copiare keywords.xlsx in data/words/

# 7. Eseguire il programma
python main.py
```

## Lista completa dei requisiti (requirements.txt)
```text
pandas>=1.3.0
PyPDF2>=2.0.0
sentence-transformers>=2.2.0
torch>=1.9.0
langchain>=0.0.200
tqdm>=4.65.0
numpy>=1.21.0
dataclasses>=0.6
typing-extensions>=4.0.0
unicodedata2>=14.0.0
openpyxl>=3.0.0
```

## Struttura Cartelle Finale
```
pdf_analyzer/
├── venv_py310/
├── data/
│   ├── input/
│   │   └── [PDF files]
│   ├── output/
│   │   └── [Generated Excel reports]
│   └── words/
│       └── keywords.xlsx
├── config.py
├── main.py
├── rag_engine.py
└── pdf_analyzer.log
```

## Troubleshooting Comune

### Windows
1. Se l'attivazione dell'ambiente virtuale fallisce:
```batch
# Provare con percorso completo
C:\Path\To\venv_py310\Scripts\activate.bat
```

2. Se appare "Execution Policy Error":
```powershell
# Aprire PowerShell come amministratore
Set-ExecutionPolicy RemoteSigned
```

### Mac/Linux
1. Se python3 non viene trovato:
```bash
# Verificare installazione
which python3

# Se necessario installare python3 (Mac)
brew install python3

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3
```

2. Se pip non viene trovato:
```bash
# Mac
brew install pip3

# Ubuntu/Debian
sudo apt-get install python3-pip
```

3. Permessi cartelle:
```bash
# Assicurarsi di avere i permessi corretti
chmod -R 755 pdf_analyzer/
```

## Note Aggiuntive

1. GPU Support (opzionale):
```bash
# Per supporto CUDA (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Per Mac M1/M2
pip install torch torchvision torchaudio
```

2. Verificare l'installazione:
```python
# In Python shell
import torch
print(torch.cuda.is_available())  # Per GPU
```

3. Memory Error:
- Aumentare la memoria swap su Linux/Mac
- Su Windows, controllare che ci sia spazio sufficiente sul disco

4. Per disattivare l'ambiente virtuale:
```bash
# Windows/Mac/Linux
deactivate
```