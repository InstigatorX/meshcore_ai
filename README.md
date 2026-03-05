# MeshCore AI Bot
### AI-powered assistant for MeshCore radio networks

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![MeshCore](https://img.shields.io/badge/MeshCore-TCP-orange)
![LLM](https://img.shields.io/badge/AI-Gemini%20%7C%20Ollama%20%7C%20Local-purple)

A lightweight Python bot that connects to a **MeshCore radio over Wi-Fi/TCP** and allows users on a mesh channel to interact with **AI models**.

Supported AI backends:

- 🌐 **Google Gemini**
- 🖥 **Ollama (local LLM)**
- 🧠 **OpenAI-compatible servers**  
  (LM Studio, Open WebUI, vLLM, llama.cpp server)

The bot is optimized for **low-bandwidth mesh networks** and automatically splits long responses into smaller packets.

---

# Example

User message on mesh:

iX-HTv4-1: !ai what is the tallest mountain?

Bot response:

@iX-HTv4-1 Mount Everest is the tallest mountain on Earth at 8,848 meters above sea level.

---

# Architecture

      Mesh Radio Network
            │
            │
    ┌───────▼────────┐
    │   MeshCore Node │
    │  TCP Interface  │
    └───────┬────────┘
            │
            │ TCP
            │
    ┌───────▼────────┐
    │   Python Bot   │
    │   mc_ai.py     │
    └───────┬────────┘
            │
   ┌────────┴───────────┐
   │                    │
   ▼                    ▼

Google Gemini       Local LLM
(Ollama /
LM Studio /
Open WebUI)

---

# Features

- Works with **MeshCore TCP interface**
- Channel-based listening (`#avl-ai`, etc.)
- Trigger commands (`!ai`)
- Replies with **@mentions**
- Handles:

!ai question
NAME: !ai question

- Duplicate packet suppression
- Response chunking for mesh packet limits
- Conversation memory
- Works with **cloud AI or fully local AI**

---

# Requirements

- Python **3.10+**
- MeshCore node with **TCP enabled**
- One of:
  - Gemini API key
  - Ollama installed
  - OpenAI-compatible LLM server

---

# Installation

## Clone the repository

git clone https://github.com/YOURUSER/meshcore-ai-bot.git
cd meshcore-ai-bot

---

# Create a Virtual Environment

Using a virtual environment prevents dependency conflicts.

python3 -m venv venv

Activate it.

Linux / macOS:

source venv/bin/activate

You should now see:

(venv)

---

# Install Dependencies

pip install –upgrade pip
pip install meshcore google-genai httpx

Optional:

pip freeze > requirements.txt

---

# Configuration

Minimum configuration:

export MESHCORE_HOST=“192.168.83.119”
export AI_TRIGGER=’!ai’

Example configuration:

export MESHCORE_HOST=“192.168.83.119”
export MESHCORE_PORT=“5000”
export MESHCORE_CHANNEL_NAME=”#avl-ai”
export AI_TRIGGER=’!ai’
export DEBUG=1

⚠️ **Important**

Always quote `!ai` in bash:

export AI_TRIGGER=’!ai’

Otherwise bash interprets `!` as history expansion.

---

# Running the Bot

python mc_ai.py

Example startup output:

[OK] Channel map:
idx=0 -> Public
idx=1 -> Loebees
idx=2 -> #avl-ai

[OK] Connected: 192.168.83.119:5000 | listening on #avl-ai

---

# Using the Bot

Test command:

!ai ping

Response:

@YourNode pong

Ask a question:

!ai why is the sky blue?

---

# AI Backend Options

The bot supports **three LLM backends**.

Select one:

export LLM_BACKEND=gemini

or

export LLM_BACKEND=ollama

or

export LLM_BACKEND=openai_compat

---

# Using Google Gemini

export LLM_BACKEND=gemini
export GEMINI_API_KEY=“YOUR_API_KEY”
export GEMINI_MODEL=“gemini-3-flash-preview”

Run:

python mc_ai.py

---

# Using Ollama (Local AI)

Install Ollama:

https://ollama.com

Pull a model:

ollama pull llama3.2

Configure the bot:

export LLM_BACKEND=ollama
export OLLAMA_BASE_URL=“http://127.0.0.1:11434”
export OLLAMA_MODEL=“llama3.2”

Run:

python mc_ai.py

---

# Using LM Studio or Other OpenAI-Compatible Servers

Example for **LM Studio**:

export LLM_BACKEND=openai_compat
export LOCAL_LLM_BASE_URL=“http://127.0.0.1:1234/v1”
export LOCAL_LLM_MODEL=“local-model”

Run:

python mc_ai.py

---

# Configuration Options

| Variable | Default | Description |
|--------|--------|--------|
| `MESHCORE_HOST` | required | MeshCore node IP |
| `MESHCORE_PORT` | 5000 | MeshCore TCP port |
| `MESHCORE_CHANNEL_NAME` | #avl-ai | Channel to listen on |
| `AI_TRIGGER` | !ai | Command prefix |
| `LLM_BACKEND` | gemini | AI backend |
| `MAX_REPLY_CHARS` | 180 | Max characters per message |
| `HISTORY_TURNS` | 6 | Conversation context |
| `DEBUG` | 0 | Enable verbose logs |

---

# Mesh Network Considerations

Mesh networks often retransmit packets.

This bot includes **duplicate suppression** to prevent multiple responses.

Some radios also **do not echo their own transmissions**.  
When testing, send commands from a **different mesh node**.

---

# Running as a Background Service

Example using **tmux**:

tmux new -s meshbot
source venv/bin/activate
python mc_ai.py

Detach:

CTRL+B D

---

# Security

Never commit API keys to GitHub.

Recommended `.gitignore`:

venv/
.env
pycache/

---

# Future Ideas

Possible enhancements:

- Weather command
- Node telemetry queries
- Emergency alert summarization
- Multi-channel AI
- RAG document search
- Voice-to-mesh gateway

---

# License

MIT License

---

# Contributing

Pull requests, ideas, and improvements are welcome.
