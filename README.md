# Story_Lab

Story_Lab is a CPU-based AI story generation system that transforms structured prompts into multi-scene narrative outputs.

##  Features

- Single Identity Prompt for character definition
- Multiple Frame Prompts for scene generation
- Configurable frame count
- LLM-powered Story Assistant (via Ollama)
- CPU-only execution (No GPU required)
- Structured multi-scene output

##  How It Works

1. User provides a character Identity Prompt.
2. User provides multiple Frame Prompts.
3. LLM assistant generates structured story prompts.
4. Diffusion pipeline generates visual story frames.
5. Final story grid is produced.

##  Tech Stack

- Python
- Gradio
- Ollama (Mistral LLM)
- Diffusion Model (DreamShaper-7)
- CPU-based inference

##  Run Locally

```bash
pip install -r requirements.txt
python app.py
