# AI Video Note Extractor

A Streamlit application that extracts video files (local or YouTube URLs), transcribes the audio, generates structured summaries/notes, and exports them as a PDF.

## 🛠️ Local Installation & Setup

To run this application locally, follow these steps:

### 1. Install System Dependency: `ffmpeg`
Both `whisper` and `moviepy` require `ffmpeg` to process audio/video files. It must be installed and added to your system's PATH.

* **Windows (via winget):**
  Open PowerShell as Administrator and run:
  ```powershell
  winget install -e --id Gyan.FFmpeg
  ```
  *Note: After installing, close and reopen VS Code to apply the system path changes.*
* **macOS (via Homebrew):**
  ```bash
  brew install ffmpeg
  ```
* **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

### 2. Install Python Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
*(If your file is named `requirement.txt` without an `s`, run `pip install -r requirement.txt` instead).*

### 3. Run the App
```bash
streamlit run app.py
```

---

## 🚀 Deployment (Streamlit Community Cloud)

1. Push your code (including `app.py`, `requirements.txt` / `requirement.txt`, and `packages.txt`) to a GitHub repository.
2. Log in to [Streamlit Share](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch, and set the entry file to `app.py`.
4. Streamlit Cloud will automatically detect `packages.txt` and install `ffmpeg` on the container, enabling audio transcription and video processing.

---

## 🔍 Troubleshooting & Common Issues

* **Memory limits on Streamlit Cloud (Exit Code 137):** 
  The models (`sshleifer/distilbart-cnn-12-6` and `whisper-base`) require substantial RAM. If the Streamlit container crashes due to memory exhaustion, try using the smaller `whisper-tiny` model or migrating the summarization logic to an API (such as OpenAI or Google Gemini).
* **Missing ffmpeg error:**
  If you get `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`, verify that `ffmpeg` is successfully installed and available in your environment's PATH.
* **YouTube Download Errors:**
  If YouTube downloads fail, update `yt-dlp` using `pip install --upgrade yt-dlp` to fetch the latest fixes for changes on YouTube.
