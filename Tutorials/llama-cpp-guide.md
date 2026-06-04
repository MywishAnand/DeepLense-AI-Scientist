# Llama.cpp Installation & Serving

## Verified Models (Tested for Agent Workflows)

The following GGUF models have been tested and confirmed to work well with llama.cpp in local, tool-enabled agent pipelines:

- `ggml-org/gpt-oss-20b-GGUF`
- `unsloth/Qwen3.5-27B-GGUF` (Recommended quantization: `Q4_K_M`)

These models are compatible with:
- Structured outputs
- Tool calling (via Jinja templates)
- Reasoning-style responses for agent workflows (e.g., DLens)

---

## Installation

### macOS
```bash
brew install llama.cpp
```

### From Source
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

## Python Bindings

```bash
pip install llama-cpp-python
pip install outlines
```

## Running a Server

### Basic Server
```bash
llama-server -m /path/to/model.gguf
```

### Example: GPT-OSS-20b
```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048
```

Server will run on: `http://127.0.0.1:8080`

### Common Parameters
- `-hf` - Hugging Face model repository
- `-m` - Local model path
- `--ctx-size` - Context size (0 for default)
- `--jinja` - Use Jinja templating
- `-ub` - Batch size (ubatch)
- `-b` - Batch size
- `-ngl` - GPU layers to offload
- `--port` - Server port (default 8080)

## Using Models from Hugging Face

### GPT OSS Model
**Repo ID:** `ggml-org/gpt-oss-20b-GGUF`  
**Filename:** `gpt-oss-20b-mxfp4.gguf`

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF -m gpt-oss-20b-mxfp4.gguf
```

## API Usage

### cURL
```bash
curl http://127.0.0.1:8080/completion -d '{
  "prompt": "Hello, world!",
  "n_predict": 128
}'
```

### Python (llama-cpp-python)
```python
from llama_cpp import Llama

llm = Llama(model_path="./model.gguf")

output = llm("Q: What is the capital of France? A:", max_tokens=32, stop=["Q:"])
print(output['choices'][0]['text'])
```

### Python (with Server)
```python
import requests

response = requests.post('http://127.0.0.1:8080/completion', 
    json={'prompt': 'Hello!', 'n_predict': 128})
print(response.json()['content'])
```

## Common Model Formats

- **GGUF** - Standard llama.cpp format
- **Q4_K_M** - 4-bit quantization (medium)
- **Q5_K_M** - 5-bit quantization (medium)
- **Q8_0** - 8-bit quantization
- **F16** - Full precision float16

## Resources

- GitHub: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- Models: [huggingface.co/models?library=gguf](https://huggingface.co/models?library=gguf)
- Python bindings: [github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
