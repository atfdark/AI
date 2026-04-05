# 🤖 Voice Assistant - Enhanced Edition

A powerful, intelligent, voice-controlled PC assistant that can automate daily computer interactions hands-free. Built with advanced speech recognition, natural language processing, and comprehensive system control capabilities.

## ✨ Features

### 🗣️ **Dual Speech Recognition Engines**
- **Google Web API**: High-accuracy online recognition
- **Vosk Offline**: Privacy-focused offline recognition
- **Auto-Fallback**: Automatic switching between engines
- **Performance Monitoring**: Real-time recognition statistics

### 🧠 **Intelligent Command Processing**
- **Intent Recognition**: Understands natural language commands
- **Confidence Scoring**: Ensures reliable command execution
- **Context Awareness**: Remembers user preferences and habits
- **Fallback Processing**: Backward compatibility with simple keywords

### 🎯 **Two Operation Modes**
- **Command Mode**: Execute system actions, open apps, control PC
- **Dictation Mode**: Convert speech to text in any application

### 🛠️ **System Control Capabilities**
- **Application Management**: Launch any configured application
- **Window Management**: Minimize, maximize, close windows
- **Text Operations**: Copy, paste, save, select all
- **Screenshot Capture**: Take screenshots with voice commands
- **Volume Control**: Adjust system volume levels
- **Web Browsing**: Open websites and perform searches

### 🔒 **Safety & Security Features**
- **Confirmation Dialogs**: Protect against dangerous operations
- **Safe File Operations**: Prevent accidental deletions
- **Secure System Control**: Confirmation for shutdown/restart

### 📊 **Performance & Monitoring**
- **Real-time Statistics**: Command success rates, recognition accuracy
- **Resource Monitoring**: CPU and memory usage tracking
- **Session Analytics**: Detailed performance reports

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd voice-assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Initial Setup

**Option A: Interactive Configuration Wizard (Recommended)**
```bash
python enhanced_launcher.py --config-wizard
```

**Option B: Manual Configuration**
Edit `config.json` to add your applications:
```json
{
  "apps": {
    "Chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "VS Code": "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "Notepad": "notepad.exe"
  }
}
```

### 3. Launch the Assistant

**Enhanced Mode (Recommended)**
```bash
python enhanced_launcher.py
```

**Classic Mode (Original)**
```bash
python enhanced_launcher.py --classic
# or
python CODE.PY
```

**Run Tests**
```bash
python enhanced_launcher.py --test
```

**Knowledge Corpus Catalog (Offline Brain Datasets)**
```bash
python enhanced_launcher.py --knowledge-catalog --knowledge-bundle starter
```

**Generate Knowledge Acquisition Plan**
```bash
python enhanced_launcher.py --knowledge-plan --knowledge-bundle starter
```

**Register and Ingest Downloaded Datasets**
```bash
python knowledge_corpus_bootstrap.py register --dataset wikipedia --path D:/datasets/wikipedia_clean
python knowledge_corpus_bootstrap.py register --dataset openstax --path D:/datasets/openstax
python knowledge_corpus_bootstrap.py ingest-registered --max-files-per-dataset 5000
```

**Show Implementation Status**
```bash
python enhanced_launcher.py --status
```

**Environment Doctor (Recommended Before First Run)**
```bash
python enhanced_launcher.py --doctor
python enhanced_launcher.py --doctor --fix
```

**Interactive Demo**
```bash
python enhanced_launcher.py --demo
```

## 📋 Voice Commands

### 🎮 **Mode Switching**
- `"start dictation"` - Begin dictation mode
- `"stop dictation"` - Return to command mode

### 📱 **Application Control**
- `"open Chrome"` - Launch Chrome browser
- `"launch VS Code"` - Start Visual Studio Code
- `"open Notepad"` - Launch Notepad
- `"close Chrome"` / `"close Notepad"` / `"close VS Code"` - Close the target app
- *(Add any application to config.json)*

### ⌨️ **App-Specific Shortcuts**
Examples supported by voice:

Chrome:
- `"new tab in chrome"`
- `"close tab in chrome"`
- `"next tab in chrome"`
- `"previous tab in chrome"`
- `"open incognito in chrome"`
- `"refresh page in chrome"`
- `"open downloads in chrome"`
- `"open history in chrome"`
- `"open developer tools in chrome"`

Notepad:
- `"new file in notepad"`
- `"open file in notepad"`
- `"save file in notepad"`
- `"save as in notepad"`
- `"find text in notepad"`
- `"replace text in notepad"`
- `"insert time date in notepad"`

VS Code:
- `"new file in vs code"`
- `"open file in vs code"`
- `"save file in vs code"`
- `"save all in vs code"`
- `"close editor in vs code"`
- `"command palette in vs code"`
- `"quick open in vs code"`
- `"open terminal in vs code"`
- `"split editor in vs code"`
- `"format document in vs code"`
- `"open settings in vs code"`
- `"open extensions in vs code"`

Tip:
- Ask `"commands for chrome"`, `"commands for notepad"`, or `"commands for vs code"` and the assistant prints full examples in the console.

### 🖥️ **System Operations**
- `"take a screenshot"` - Capture screen
- `"close window"` - Close active window
- `"increase volume"` - Turn up volume
- `"decrease volume"` - Turn down volume

### ✍️ **Text Operations**
- `"copy"` - Copy selected text
- `"paste"` - Paste from clipboard
- `"save"` - Save current document
- `"select all"` - Select all text

### 🔍 **Web & Search**
- `"search for Python tutorials"` - Google search
- `"open github.com"` - Visit website
- `"go to youtube"` - Navigate to site

### 🗣️ **Natural Language**
The assistant understands various phrasings:
- `"I want to open Chrome"`
- `"Please take a screenshot"`
- `"Can you increase the volume?"`
- `"Open VS Code please"`

## ⚙️ Configuration

### Speech Recognition Settings

Add to `config.json`:
```json
{
  "speech_recognition": {
    "preferred_engine": "auto",  // "google", "vosk", or "auto"
    "vosk_model_path": "vosk-model-small-en-us-0.15",
    "energy_threshold": 300,
    "dynamic_energy_threshold": true
  }
}
```

### Safety Settings

```json
{
  "safety": {
    "confirm_delete": true,
    "confirm_shutdown": true,
    "confirm_website": false
  }
}
```

### Voice Settings

```json
{
  "tts": {
    "engine": "edge_tts",
    "edge_tts": {
      "voice": "en-US-GuyNeural",
      "rate": "+0%",
      "pitch": "+0Hz",
      "volume": "+0%",
      "output_format": "riff-24khz-16bit-mono-pcm"
    }
  }
}
```

### Cloud LLM Fallback (Claude / Gemini)

```json
{
  "cloud_llm": {
    "enabled": false,
    "provider": "anthropic",
    "anthropic_api_key": "",
    "anthropic_model": "claude-3-5-sonnet-latest",
    "gemini_api_key": "",
    "gemini_model": "gemini-1.5-flash"
  }
}
```

Notes:
- Keep `enabled: false` until you add an API key.
- If Ollama is enabled, local answering is attempted first; cloud is used as fallback.

## 🔧 Advanced Features

### Implementation Status Report

Use the launcher status mode to quickly see what is implemented in your current setup:

```bash
python enhanced_launcher.py --status
```

The report includes:
- Feature groups implemented in `assistant/`
- Current configuration snapshot (apps, wake word, language, preferred engine)
- Dependency availability (speech, NLP, ML, Windows integrations)
- Test script coverage snapshot

### Offline Speech Recognition (Vosk)

1. **Install Vosk dependencies**: Already included in requirements.txt
2. **Download model**: Use config wizard or manual download
3. **Configure**: Set `preferred_engine` to "vosk" or "auto"

**Model Options:**
- **Small English** (50MB): `vosk-model-small-en-us-0.15.zip`
- **Large English** (1.8GB): `vosk-model-en-us-0.22.zip`

### Performance Monitoring

Press these keys during runtime:
- **`s`** - Show current status
- **`r`** - Show recognition statistics  
- **`q`** - Quit assistant

### Learning & Personalization

The assistant learns from your usage patterns:
- **Frequent Commands**: Remembers your most-used commands
- **Custom Commands**: Adapts to your terminology
- **Performance Optimization**: Improves accuracy over time

## 🧪 Testing & Troubleshooting

### Run Comprehensive Tests
```bash
python enhanced_launcher.py --test
```

### Test Individual Components
```bash
python test_assistant.py
```

### Common Issues & Solutions

**Microphone not working:**
1. Check microphone permissions in system settings
2. Verify microphone is not muted
3. Run: `python -c "import speech_recognition; print('OK')"`

**Speech recognition fails:**
1. Check internet connection (for Google API)
2. Verify microphone calibration: `python -c "import speech_recognition; r=speech_recognition.Recognizer(); m=speech_recognition.Microphone(); with m as source: r.adjust_for_ambient_noise(source); print('OK')"`

**Applications won't launch:**
1. Verify paths in `config.json` are correct
2. Use full absolute paths
3. Test paths manually in file explorer

**TTS not working:**
1. Check system audio settings
2. Verify text-to-speech engine is installed
3. Try running: `python -c "import pyttsx3; engine=pyttsx3.init(); engine.say('test'); engine.runAndWait()"`

## 🏗️ Architecture

```
Voice Assistant
├── 🎤 Speech Recognition
│   ├── Google Web API (online)
│   ├── Vosk Offline Engine
│   └── Auto-fallback system
├── 🧠 Command Parser
│   ├── Intent Recognition
│   ├── Natural Language Processing
│   └── Confidence Scoring
├── 🛠️ Action Executor
│   ├── Application Control
│   ├── System Operations
│   └── Text Manipulation
└── 🔊 Text-to-Speech
    ├── Voice Feedback
    ├── Command Confirmation
    └── Error Notifications
```

## 📁 File Structure

```
voice-assistant/
├── enhanced_launcher.py          # Main launcher
├── config_wizard.py              # Interactive setup
├── test_assistant.py             # Test suite
├── CODE.PY                       # Classic launcher
├── config.json                   # Configuration
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── assistant/
    ├── main.py                   # Classic main
    ├── main_enhanced.py          # Enhanced main
    ├── speech.py                 # Classic speech
    ├── speech_enhanced.py        # Enhanced speech
    ├── parser.py                 # Classic parser
    ├── parser_enhanced.py        # Enhanced parser
    ├── tts.py                    # Text-to-Speech
    ├── actions.py                # System actions
    └── __init__.py
```

## 🔮 Future Enhancements

- **Machine Learning Integration**: Custom model training
- **Multi-language Support**: Recognition in multiple languages
- **Advanced Web Automation**: Complex web interactions
- **Email Integration**: Send/receive emails by voice
- **Calendar Management**: Voice-controlled scheduling
- **Smart Home Integration**: IoT device control
- **Plugin System**: Extensible command architecture

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional speech recognition engines
- New command categories
- Platform-specific optimizations
- UI/UX improvements
- Performance optimizations

## 📄 License

This project is open source. Feel free to modify and distribute.

## ⚡ Quick Reference

| Task | Command |
|------|---------|
| Start Assistant | `python enhanced_launcher.py` |
| Configuration Wizard | `python enhanced_launcher.py --config-wizard` |
| Run Tests | `python enhanced_launcher.py --test` |
| Implementation Status | `python enhanced_launcher.py --status` |
| Interactive Demo | `python enhanced_launcher.py --demo` |
| Classic Mode | `python enhanced_launcher.py --classic` |
| Help | `python enhanced_launcher.py --help` |

---

**🎉 Your intelligent voice assistant is ready to transform how you interact with your computer!**

For support or questions, check the troubleshooting section above or run the interactive demo to explore all features.
