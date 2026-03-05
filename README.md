:::writing{variant=“standard” id=“48261”}

MeshCore Gemini AI Bot

A lightweight Python bot that connects to a MeshCore radio over Wi-Fi/TCP and allows users on a mesh channel to interact with Google Gemini AI.

Users can send commands like:

!ai where is Paris?

The bot will respond in-channel and mention the user:

@iX-HTv4-1 Paris is the capital of France, located in north-central France along the Seine River.

The bot is optimized for low-bandwidth mesh networks, keeping responses concise and splitting longer replies into multiple packets.

⸻

Features
	•	Works with MeshCore TCP interface
	•	Channel-based listening (ex: #avl-ai)
	•	Trigger commands (!ai)
	•	Responds with @mentions to the sender
	•	Handles messages formatted like:

!ai question
NAME: !ai question

	•	De-duplicates repeated packets (common on mesh networks)
	•	Splits long AI responses into multiple mesh messages
	•	Maintains short conversation context for better answers
	•	Fully configurable via environment variables

⸻

Example

User message:

iX-HTv4-1: !ai what is the tallest mountain?

Bot response:

@iX-HTv4-1 Mount Everest is the tallest mountain on Earth at 8,848 meters above sea level.


⸻

Requirements
	•	Python 3.10+
	•	A MeshCore node with TCP enabled
	•	A Google Gemini API key

⸻

Installation

1. Clone the repository

git clone https://github.com/YOURNAME/meshcore-gemini-bot.git
cd meshcore-gemini-bot


⸻

Create a Python Virtual Environment

Using a virtual environment keeps dependencies isolated from your system Python.

Create the venv

python3 -m venv venv

This creates:

meshcore-gemini-bot/
  venv/
  mc_ai.py

Activate the venv

Linux / macOS:

source venv/bin/activate

Your prompt should now show:

(venv)


⸻

Install Dependencies

pip install --upgrade pip
pip install meshcore google-genai

Optional: save dependencies

pip freeze > requirements.txt


⸻

Configure Environment Variables

Set the required variables before running the bot.

export GEMINI_API_KEY="YOUR_API_KEY"
export MESHCORE_HOST="192.168.1.100"

Optional configuration:

export MESHCORE_PORT="5000"
export MESHCORE_CHANNEL_NAME="#avl-ai"
export AI_TRIGGER='!ai'
export DEBUG=1

Important

When setting !ai in bash, use single quotes:

export AI_TRIGGER='!ai'

Otherwise bash will interpret ! as a history expansion.

⸻

Running the Bot

Start the bot:

python mc_ai.py

Example output:

[OK] Channel map:
  idx=0 -> Public
  idx=1 -> Loebees
  idx=2 -> #avl-ai

[OK] Connected: 192.168.83.119:5000 | listening on #avl-ai (idx=2)


⸻

Using the Bot

Send commands in the configured mesh channel.

Example:

!ai ping

Response:

@YourNode pong

Ask a question:

!ai what causes lightning?

Bot response:

@YourNode Lightning forms when electrical charge builds up in clouds and discharges toward the ground or another cloud.


⸻

Configuration

Variable	Default	Description
GEMINI_API_KEY	required	Google Gemini API key
MESHCORE_HOST	required	MeshCore node IP
MESHCORE_PORT	5000	MeshCore TCP port
MESHCORE_CHANNEL_NAME	#avl-ai	Channel to listen on
AI_TRIGGER	!ai	Command prefix
MAX_REPLY_CHARS	180	Max characters per mesh message
HISTORY_TURNS	6	AI conversation context length
DEBUG	0	Enable verbose logging


⸻

Notes About Mesh Networks

Mesh protocols often retransmit packets. This bot includes duplicate suppression so it won’t reply multiple times to the same message.

Additionally, some radios do not echo messages sent from the same node. If testing the bot, send commands from a different mesh node.

⸻

Security

Never commit your API key.

Use environment variables or a .env file ignored by Git.

Example .gitignore entry:

.env
venv/


⸻

Running as a Background Service (Optional)

Example using tmux:

tmux new -s meshbot
source venv/bin/activate
python mc_ai.py

Detach:

CTRL+B D


⸻

Future Ideas
	•	Weather commands
	•	Node status queries
	•	Mesh telemetry summaries
	•	Local LLM support
	•	Store-and-forward summaries
	•	Multi-channel support

⸻

License

MIT License

⸻

Contributing

Pull requests and improvements are welcome.
:::
