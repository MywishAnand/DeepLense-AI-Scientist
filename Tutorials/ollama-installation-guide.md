# Ollama Installation & Serving

## Installation

### macOS/Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download installer from [ollama.com/download](https://ollama.com/download)

### Docker
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Serving

```bash
ollama serve
```

Server runs on `http://localhost:11434`

### Environment Variables
```bash
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=/path/to/models
```

## Basic Usage

```bash
# Pull a model
ollama pull llama2

# Run interactively
ollama run llama2

# List models
ollama list

# Remove model
ollama rm llama2
```

## API

### cURL
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello!"
}'
```

### Python
```python
import requests

response = requests.post('http://localhost:11434/api/generate', 
    json={'model': 'llama2', 'prompt': 'Hello!', 'stream': False})
print(response.json()['response'])
```

## Custom Models

Create `Modelfile`:
```dockerfile
FROM llama2
PARAMETER temperature 0.8
SYSTEM You are a helpful assistant.
```

```bash
ollama create my-model -f Modelfile
ollama run my-model
```
