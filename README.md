# AI Video Note Extractor

AI-powered application that converts YouTube videos or uploaded videos into structured study notes and downloadable PDF summaries.

## Features

- Upload local video files
- Process YouTube video links
- Extract audio automatically
- Speech-to-text transcription using OpenAI Whisper
- AI-powered summarization using Transformer models
- Generate study notes and revision points
- Export notes as PDF

## Technologies Used

- Python
- Streamlit
- OpenAI Whisper
- Transformers (Hugging Face)
- MoviePy
- yt-dlp
- FPDF

## How It Works

1. Upload a video or provide a YouTube link
2. Audio is extracted from the video
3. Whisper converts speech into text
4. Transformer model generates summarized notes
5. Notes are exported as a downloadable PDF

## Installation

```bash
pip install -r requirements.txt
