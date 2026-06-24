# musemender

Upload any song and find the closest matches from your own music library using CLAP audio embeddings + FAISS vector search.

## Requirements

- Ubuntu 20.04 / 22.04 / 24.04
- Python 3.10 or higher
- ~4GB free disk space (CLAP model ~600MB + your library vectors)
- ~6GB RAM recommended (CLAP loads ~600MB into memory)

---

## Setup (one time)

```bash
# 1. Clone or copy the musemender folder to your machine, then enter it
cd musemender

# 2. Run the installer
bash install.sh
```

The installer will:
- Check your Python version (3.10+ required)
- Install `ffmpeg` and `python3-venv` via apt
- Create a `venv/` virtual environment
- Install all Python packages
- Verify every import works

---

## Usage

### Step 1 — Index your music library

```bash
bash index.sh /path/to/your/Music
```

**First run only:** downloads the CLAP model (~600MB). Each song takes 15–45s on CPU.  
Run overnight for large libraries.

```bash
# After adding new songs — only re-indexes changed/new files
bash index.sh /path/to/your/Music --update

# Remove deleted songs from the index
bash index.sh /path/to/your/Music --clean
```

### Step 2 — Start the app

```bash
bash start.sh
```

### Step 3 — Open in browser

```
http://localhost:5000
```

---

## Project structure

```
musemender/
├── backend/
│   ├── app.py          # Flask server + API endpoints
│   ├── config.py       # All paths and constants (edit here for customisation)
│   ├── db.py           # SQLite operations
│   ├── extractor.py    # CLAP embedding pipeline
│   ├── indexer.py      # Bulk library indexer
│   └── similarity.py   # FAISS search
├── frontend/
│   └── index.html      # Single-file web UI
├── data/               # Created at runtime
│   ├── library.db      # Song metadata (SQLite)
│   ├── vectors.index   # FAISS vector index
│   └── id_map.json     # FAISS row → DB ID mapping
├── uploads/            # Temp files during upload (auto-cleaned)
├── venv/               # Python virtual environment (created by install.sh)
├── install.sh          # One-shot installer
├── start.sh            # Start the web server
├── index.sh            # Index your music library
└── requirements.txt    # Python dependencies
```

---

## Supported audio formats

`.mp3` `.wav` `.flac` `.ogg` `.m4a` `.aac` `.wma`

> `.wma` `.aac` `.m4a` require `ffmpeg` for conversion (installed automatically).

---

## Configuration

All tuneable values are in `backend/config.py`:

| Setting | Default | Description |
|---|---|---|
| `FLASK_PORT` | `5000` | Web server port |
| `MAX_UPLOAD_BYTES` | `50MB` | Upload size limit |
| `MAX_DURATION_SECONDS` | `300` | Cap audio at 5 min for embedding |
| `MIN_DURATION_SECONDS` | `5` | Reject clips shorter than this |
| `DEFAULT_TOP_N` | `10` | Default number of results |
| `LOW_CONFIDENCE_THRESHOLD` | `0.60` | Warn if best match is below this score |

---

## VS Code

Open the folder in VS Code and select the venv interpreter:

```
Ctrl+Shift+P → Python: Select Interpreter → ./venv/bin/python
```

Launch configs in the Run & Debug tab:
- **▶ Start Flask Server**
- **🎵 Index Library (full / --update / --clean)**
- **🔬 Test Extractor** — verify a single file embeds correctly

---

## Troubleshooting

**`Cannot reach backend` in the browser**  
→ Make sure `bash start.sh` is running and check the terminal for errors.

**`No index found`**  
→ Run `bash index.sh /path/to/Music` before starting the server.

**Song takes forever to embed**  
→ Normal on CPU — CLAP takes 15–45s per song. Use `--update` for incremental runs.

**`.wma` / `.aac` files fail**  
→ `ffmpeg` must be installed: `sudo apt install ffmpeg`

**`Python 3.10+ required` error on Ubuntu 20.04**  
→ `sudo apt install python3.10 python3.10-venv` then re-run `install.sh`

**Low similarity scores across the board**  
→ Lower `LOW_CONFIDENCE_THRESHOLD` in `config.py`, or your library may genuinely be diverse.
