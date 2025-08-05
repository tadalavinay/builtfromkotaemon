# OpenRouter Chatbot with Web Interface

A powerful, production-ready chatbot implementation that leverages OpenRouter API for chat completions and OpenAI for embeddings. Features both command-line and web interfaces with document upload capabilities.

## 🚀 Features

- **Multiple Interfaces**: Command-line and web-based chat interfaces
- **OpenRouter Integration**: Uses OpenRouter API for cost-effective chat completions
- **Document Processing**: Upload and chat with PDF, TXT, and other documents
- **Real-time Web Interface**: Modern, responsive web chat with typing indicators
- **Streaming Responses**: Real-time message streaming in web interface
- **Error Handling**: Graceful handling of API errors and rate limits
- **Demo Mode**: Run without API keys for testing purposes
- **Comprehensive Testing**: Full test suite included

## 📋 Prerequisites

- Python 3.8+
- OpenRouter API key (get one at [openrouter.ai](https://openrouter.ai))
- OpenAI API key (for embeddings, get one at [openai.com](https://openai.com))

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd openrouter-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
```

## 🎯 Quick Start

### Web Interface (Recommended)
```bash
# Run the web chatbot
python run_web_chatbot.py

# Access at http://localhost:5001
```

### Command Line Interface
```bash
# Run simple chatbot
python run_chatbot.py

# Run OpenRouter chatbot
python run_openrouter_chatbot.py

# Run demo (no API keys needed)
python run_demo_web_chatbot.py
```

## 📁 Project Structure

```
├── chatbot_app.py           # Core chatbot application
├── web_chatbot.py          # Flask web interface
├── openrouter_chatbot.py   # OpenRouter integration
├── document_loader.py      # Document processing
├── templates/
│   └── chat.html          # Web interface template
├── tests/
│   ├── test_chatbot.py
│   ├── test_openrouter.py
│   └── test_web_chatbot.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENAI_API_KEY`: Your OpenAI API key (for embeddings)
- `PORT`: Web server port (default: 5001)

### Supported Models
- **OpenRouter Models**: Any model available through OpenRouter
- **Default**: `openai/gpt-3.5-turbo` (cost-effective)
- **Premium**: `openai/gpt-4`, `anthropic/claude-3-opus`

## 🧪 Testing

Run the complete test suite:
```bash
python -m pytest test_*.py -v
```

Individual test files:
```bash
python test_chatbot.py
python test_openrouter.py
python test_web_chatbot.py
```

## 🌐 Web Interface Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Chat**: Instant responses with typing indicators
- **File Upload**: Drag-and-drop document upload
- **Message History**: Persistent chat history
- **Error Handling**: User-friendly error messages
- **Health Check**: Built-in health monitoring

## 📊 API Endpoints

### Web Interface
- `GET /` - Main chat interface
- `POST /chat` - Send message and get response
- `POST /chat_stream` - Streaming chat endpoint
- `POST /upload` - Upload documents
- `GET /health` - Health check

## 🔍 Usage Examples

### Basic Chat
```python
from openrouter_chatbot import OpenRouterChatbot

chatbot = OpenRouterChatbot(api_key="your_key")
response = chatbot.chat("Hello, how are you?")
print(response)
```

### Document Chat
```python
from chatbot_app import ChatbotApp

app = ChatbotApp()
app.load_document("document.pdf")
response = app.chat("What is this document about?")
```

### Web Integration
```python
from web_chatbot import create_app

app = create_app()
app.run(host='0.0.0.0', port=5001)
```

## 🚨 Troubleshooting

### Common Issues

1. **Port 5000 conflicts on macOS**
   - Solution: Uses port 5001 by default

2. **OpenRouter API errors**
   - Check your API key validity
   - Verify model availability

3. **Document upload fails**
   - Ensure file format is supported (PDF, TXT, etc.)
   - Check file size limits

4. **Memory issues with large documents**
   - Increase system memory
   - Process documents in chunks

### Debug Mode
```bash
# Enable debug logging
export FLASK_ENV=development
python run_web_chatbot.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai) for providing the API
- [OpenAI](https://openai.com) for embeddings API
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [LangChain](https://langchain.com) for document processing

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information