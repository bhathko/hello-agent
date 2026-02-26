# Hello-Agent 🤖

A modular AI Agent built with the modern **Google Gemini SDK** and the **ReAct (Reasoning + Acting)** pattern. This agent can check weather information via Google Maps Platform and recommend travel attractions using Tavily Search.

## 🌟 Features
- **Modern Gemini SDK**: Uses `google-genai` for high-performance LLM interactions.
- **Google Weather Integration**: Real-time hyperlocal weather data via Google Maps Platform.
- **Smart Recommendations**: Attraction searching powered by Tavily.
- **Modular Structure**: Clean, package-based architecture for better maintainability.
- **ReAct Pattern**: Transparent thought process and tool usage.

## 📂 Project Structure
```text
/
├── base_chapter.py                  # Abstract base class for all chapters
├── chapter_1_transformer_structure/ # Chapter 1: Transformer Architecture
│   ├── run.py                       # Self-contained entry point for Chapter 1
│   └── simple_transformer.py        # Skeleton of Transformer implementation
├── chapter_2_action_thought_observe/# Chapter 2: ReAct Agent Implementation
│   ├── run.py                       # Self-contained entry point for Chapter 2
│   ├── agent.py                     # ReAct agent logic (Thought-Action-Observation)
│   ├── config.py                    # Configuration & Validation
│   ├── llm_client.py                # Gemini API client
│   ├── prompts.py                   # System instructions
│   └── tools/                       # Tool implementations
│       ├── weather.py               # Google Weather API tool
│       └── attraction.py            # Tavily Search tool
├── main.py                          # Global entry point (switching between chapters)
├── .env                             # Environment variables (private)
└── requirements.txt                 # Dependencies
```

## 🚀 Setup

### 1. Prerequisites
- Python 3.9+
- A virtual environment (`python -m venv .venv`)

### 2. Install Dependencies
```bash
source .venv/bin/activate  # Or your platform equivalent
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
# Gemini (Get at aistudio.google.com)
GEMINI_API_KEY=your_gemini_key
MODEL_ID=gemini-1.5-flash

# Google Maps Platform (Enable Geocoding & Weather API)
GOOGLE_API_KEY=your_google_maps_key

# Tavily (Get at tavily.com)
TAVILY_API_KEY=your_tavily_key
```

## ⚠️ Important Note on Weather API
The **Google Maps Platform Weather API** is highly accurate but currently has **regional limitations**. 
- **Supported**: Most international cities (London, New York, Tokyo, etc.).
- **Known Limitations**: Currently does not support some regions like Beijing or other parts of mainland China (returns 404).
- **Behavior**: The agent is programmed to recognize these limitations and will inform you if a city is not yet supported.

## 🛠 Usage
You can run the global entry point:

```bash
# Run Chapter 1
python main.py --chapter 1

# Run Chapter 2
python main.py --chapter 2
```

Or run each chapter's entry point directly:

```bash
# Chapter 1
python chapter_1_transformer_structure/run.py

# Chapter 2
python chapter_2_action_thought_observe/run.py
```
