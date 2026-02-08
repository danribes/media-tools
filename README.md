# Media Tools

A command-line toolkit for downloading, converting, and subtitling videos. Features smart auto-processing with Whisper speech recognition, OCR subtitle detection, and automatic translation.

## Features

- **Download** videos from URLs via yt-dlp
- **Convert** to 10:9 aspect ratio (pixel-stretch with H.264 re-encode)
- **Generate subtitles** with burned-in text:
  - Audio transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  - OCR detection of existing burned-in subtitles (Tesseract)
  - Auto-translation via Google Translate (deep-translator)
  - Smart placement — detects existing text regions and places new subs in free space
- **Auto-process** — a single command that assesses the video and decides what to do:
  - Detects aspect ratio, audio language, and burned-in subtitles
  - Applies the right pipeline automatically (convert, transcribe, translate, or skip)
  - Parallel assessment, single-pass ffmpeg encoding, GPU auto-detection

## Requirements

- Python 3.10+
- ffmpeg / ffprobe (place in `bin/` or have on PATH)
- Tesseract OCR (`sudo apt install tesseract-ocr`)

### Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install faster-whisper deep-translator pytesseract pillow numpy yt-dlp
```

Optional for GPU acceleration:

```bash
pip install torch  # CUDA auto-detected for Whisper inference
```

## Usage

### Interactive menu

```bash
./media-tools.sh
```

```
╔══════════════════════════════════════╗
║           MEDIA TOOLS                ║
╠══════════════════════════════════════╣
║  1) Download video                   ║
║  2) Convert to 10:9                  ║
║  3) Generate subtitles               ║
║  4) Download + Convert to 10:9       ║
║  5) Download + Subtitle              ║
║  6) Download + Convert + Subtitle    ║
║  7) Auto-process (smart)             ║
║  8) Quit                             ║
╚══════════════════════════════════════╝
```

### Individual scripts

```bash
# Download a video
./download.sh "https://example.com/video"

# Convert to 10:9 aspect ratio
./convert109.sh input.mp4

# Generate and burn subtitles (auto-detect language, translate to Spanish)
./subtitles.sh input.mp4

# Generate English subtitles, skip OCR scan
./subtitles.sh input.mp4 --target-lang en --no-ocr
```

### Auto-process (smart mode)

Automatically assesses the video and runs only the steps needed:

```bash
# Activate venv first
source venv/bin/activate

# Auto-process a local file
python3 scripts/auto_process.py input.mp4

# Auto-process a URL (downloads first)
python3 scripts/auto_process.py "https://example.com/video"

# Dry run — assess and show plan without executing
python3 scripts/auto_process.py input.mp4 --dry-run

# Target English subtitles instead of Spanish
python3 scripts/auto_process.py input.mp4 --target-lang en
```

#### Decision matrix

| Audio Language | Burned-in Subs | Action |
|---|---|---|
| Target (e.g. Spanish) | Yes | Skip subtitles |
| Target | No | Transcribe only (no translation) |
| Other (e.g. English) | Yes | OCR sync + translate |
| Other | No | Whisper transcribe + translate |

Aspect ratio conversion to 10:9 is added automatically when needed.

## Project structure

```
media-tools/
├── media-tools.sh          # Interactive menu (entry point)
├── download.sh             # Download videos via yt-dlp
├── convert109.sh           # Convert to 10:9 aspect ratio
├── subtitles.sh            # Generate + burn subtitles (wrapper)
├── download-and-convert.sh # Download + convert combo
├── scripts/
│   ├── subtitle_gen.py     # Subtitle engine (Whisper, OCR, translation, ASS/SRT)
│   └── auto_process.py     # Smart auto-processing pipeline
├── bin/                    # ffmpeg + ffprobe binaries (not in repo)
├── venv/                   # Python virtual environment (not in repo)
├── downloads/              # Downloaded videos (not in repo)
├── converted/              # 10:9 converted videos (not in repo)
└── subbed/                 # Subtitled output videos (not in repo)
```

## License

MIT
