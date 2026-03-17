# GPT-TranscriberUI

Interfaccia grafica (CustomTkinter) per dividere file audio in chunk e trascriverli con l'API di OpenAI.

## Requisiti

- **Python 3.10+**
- **ffmpeg** installato nel sistema (necessario per caricare/esportare audio con pydub)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - Windows: `choco install ffmpeg` (oppure scarica binari e aggiungi alla variabile PATH)

## Installazione rapida

1. Clona la repo e posizionati nella cartella.
2. (Opzionale) Crea e attiva un virtual environment.
3. **Esegui lo starter**:
   ```bash
   python3 start.py
   ```
   Lo script:
   - aggiorna `pip`,
   - installa i pacchetti mancanti da `requirements.txt`,
   - verifica la presenza di `ffmpeg`,
   - lancia la GUI (`main_ui.py`).

In alternativa, installa manualmente:
```bash
pip install -r requirements.txt
python3 main_ui.py
```

## Configurazione chiave API

Crea un file `.env` nella root del progetto con:
```
OPENAI_API_KEY=la_tua_chiave
```
La GUI può caricare la chiave da `.env` (bottone “Carica da .env”).  
Il codice usa `python-dotenv` per caricare la variabile (vedi `main_ui.py`).

## Utilizzo

1. Avvia l'app con `python3 start.py` (o `python3 main_ui.py`).
2. Seleziona un file audio (mp3/m4a/wav/ogg/flac/webm).
3. Scegli il percorso di output del testo.
4. (Facoltativo) Inserisci **Prompt** per indirizzare la trascrizione.
5. (Facoltativo) Cambia il modello (es. `whisper-1` consigliato).
6. Premi **“Avvia Trascrizione”**.

L'app dividerà l'audio in `audio_chunks/` e trascriverà i pezzi in sequenza.

## Note tecniche

- La UI è in `main_ui.py` e importa:
  - `customtkinter` per l’interfaccia
  - `python-dotenv` per caricare la chiave da `.env`
  - funzioni locali `split_audio.py` e `transcribe2.py`
- La suddivisione audio usa **pydub** (richiede `ffmpeg`).
- La trascrizione usa l’SDK `openai`. Se il modello selezionato non è disponibile, viene suggerito di provare `whisper-1`.

## Troubleshooting

- **“ffmpeg non trovato”**: assicurati che `ffmpeg` sia installato e presente nel PATH.  
- **Errore modello**: se compare un errore “model not found”, usa `whisper-1`.
- **Linux**: potrebbe essere necessario installare `python3-tk` per tkinter/CustomTkinter.

## Licenza

TBD
