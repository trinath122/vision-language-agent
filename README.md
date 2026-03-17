# Vision-Language Agentic Reasoning System

A multimodal AI agent that combines image understanding with large language model reasoning. The system analyzes visual inputs, extracts meaningful features, and uses an LLM-powered agentic loop to answer natural language questions about images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Components In Depth](#components-in-depth)
5. [Setup & Installation](#setup--installation)
6. [Running the Agent](#running-the-agent)
7. [Configuration](#configuration)
8. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
9. [Hardware Constraints & Adaptations](#hardware-constraints--adaptations)
10. [Future Work](#future-work)

---

## Project Overview

This project implements a **Vision-Language Agent** — a system that can:

- Accept a natural language query and an optional image
- Analyze the image using computer vision techniques
- Reason about the image content using a language model
- Return a grounded, coherent response

The agent is built using a **tool-augmented reasoning loop** powered by [LangGraph](https://github.com/langchain-ai/langgraph), with image analysis handled by PIL (Pillow) and the language backbone served locally via [Ollama](https://ollama.ai) running `deepseek-r1:1.5b`.

The model architecture is designed around the concept of **visual token injection** — encoding image features into a representation that can be prepended to language model inputs, similar to how models like LLaVA and InstructBLIP work.

---

## Architecture

```
User Query + Image Path
        |
        v
  ┌─────────────────────┐
  │   CLI Entry Point   │  main.py
  │   (argparse)        │
  └────────┬────────────┘
           |
           v
  ┌─────────────────────────────────────────────┐
  │              LangGraph Agent Loop            │
  │                                             │
  │   ┌──────────┐     ┌──────────────────┐    │
  │   │  Agent   │────>│  should_continue │    │
  │   │  Node    │     │  (stop condition)│    │
  │   └────┬─────┘     └────────┬─────────┘    │
  │        |                    |               │
  │        v                    v               │
  │   ┌──────────┐           END / Tools        │
  │   │  Tools   │                              │
  │   │  Node    │                              │
  │   └──────────┘                              │
  └─────────────────────────────────────────────┘
           |
           v
  ┌─────────────────────┐
  │  Image Analyzer     │  PIL-based feature extraction
  │  - Dominant color   │  (color, brightness, resolution)
  │  - Brightness       │
  │  - Resolution       │
  └────────┬────────────┘
           |
           v
  ┌─────────────────────┐
  │  Ollama LLM         │  deepseek-r1:1.5b (local)
  │  deepseek-r1:1.5b   │  Receives image description
  │                     │  + user query as prompt
  └─────────────────────┘
           |
           v
     Agent Response
```

### Core Architecture: Visual Token Injection

The model architecture (`src/models/`) implements the full vision-language fusion pipeline intended for production use:

```
Image
  |
  v
CLIP Vision Encoder (ViT-B/32 or ViT-L/14)
  |
  v
Vision Projection Layer (MLP: clip_dim → llm_dim)
  |
  v
Visual Tokens [v1, v2, ..., vk]
  |
  v
Concatenate with Text Token Embeddings
  |
  [v1, v2, ..., vk, t1, t2, ..., tn]
  |
  v
Language Model (Causal LM decoder)
  |
  v
Generated Response
```

This is the same approach used by LLaVA (Liu et al., 2023) and InstructBLIP, where visual features are projected into the LLM's embedding space and prepended to the text sequence.

---

## Project Structure

```
vision-lang-agent/
│
├── main.py                         # CLI entry point
├── requirements.txt                # All Python dependencies
├── configs/
│   └── config.yaml                 # Model, training, and agent configuration
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── reasoning_agent.py      # LangGraph agent loop, image analysis, LLM calls
│   │   └── tools.py                # Agent tools: captioning, VQA, detection, grounding
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_encoder.py         # CLIP vision encoder + vision projection layer
│   │   └── multimodal_llm.py       # Full VisionLanguageModel (CLIP + LLM fusion)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py             # Pretraining loop with contrastive + generative loss
│   │   └── dpo.py                  # DPO (Direct Preference Optimization) fine-tuning
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py             # RAGAS / DeepEval evaluation pipeline
│   │
│   └── utils/
│       ├── __init__.py
│       └── data_utils.py           # Dataset loading and preprocessing utilities
│
├── notebooks/                      # Jupyter notebooks for exploration
├── data/                           # Dataset storage (gitignored)
├── scripts/                        # Training and deployment scripts
├── tests/                          # Unit tests
├── deployment/
│   └── gcp_deploy.py               # GCP Vertex AI deployment script
└── .env.example                    # Environment variable template
```

---

## Components In Depth

### 1. CLIP Vision Encoder (`src/models/clip_encoder.py`)

The `CLIPVisionEncoder` wraps OpenCLIP's pretrained ViT-B/32 or ViT-L/14 model. It:

- Encodes input images into normalized embedding vectors (shape: `[B, embed_dim]`)
- Freezes all CLIP weights during training — only the projection layer learns
- Also supports text encoding for contrastive alignment tasks

```python
class CLIPVisionEncoder(nn.Module):
    def encode_image(self, images) -> torch.Tensor:
        # Returns normalized image embeddings: (B, embed_dim)
```

**Why freeze CLIP?** CLIP is already trained on 400M image-text pairs. Freezing it preserves this strong visual representation while keeping compute cost low — only the small projection MLP needs to be trained.

---

### 2. Vision Projection Layer (`src/models/clip_encoder.py`)

The `VisionProjection` bridges the vision encoder and the language model:

```
CLIP output: (B, 512)  →  MLP  →  (B, num_patches, llm_dim)
```

It uses a two-layer MLP with GELU activation:

```
Linear(clip_dim → llm_dim * 2) → GELU → Linear(llm_dim * 2 → llm_dim * num_patches)
```

The output is reshaped to `(B, num_patches, llm_dim)` to produce one or more "visual tokens" that are prepended to the text embedding sequence before the language model processes them.

---

### 3. VisionLanguageModel (`src/models/multimodal_llm.py`)

The full fusion model combines CLIP encoder + projection + a causal language model:

**Forward pass:**
1. Text tokens → LLM embedding layer → text embeddings `(B, T, D)`
2. Image → CLIP → projection → visual tokens `(B, P, D)`
3. Concatenate: `[visual_tokens | text_embeddings]` → `(B, P+T, D)`
4. Pass through LLM decoder → output logits

**LoRA support:** The model optionally applies LoRA (Low-Rank Adaptation) to the LLM's attention matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`) for parameter-efficient fine-tuning. This reduces trainable parameters by ~99% compared to full fine-tuning.

**INT8 Quantization:** Supports loading the LLM in 8-bit quantization via `bitsandbytes`, reducing VRAM from ~14GB to ~7GB for a 7B parameter model.

---

### 4. LangGraph Agent Loop (`src/agents/reasoning_agent.py`)

The agent is built as a **stateful graph** using LangGraph:

```
START → [agent node] → should_continue? → END
                              |
                         [tools node] → back to [agent node]
```

**Agent State:**
```python
class AgentState(TypedDict):
    messages: list          # Full conversation history
    image_path: str | None  # Path to input image
    iteration: int          # Current reasoning step
    max_iterations: int     # Max steps before forced stop
```

**Reasoning loop:**
1. `call_model` — If an image is provided, analyze it first, then build a prompt combining image description + conversation history + system prompt, and call the LLM
2. `should_continue` — Stop if max iterations reached or if the model returns a final answer (no tool calls)
3. `call_tools` — Execute any tool calls the agent requested

**Image analysis (PIL-based):**
```python
def describe_image_with_pil(image_path):
    # Extracts: resolution, dominant RGB color, brightness level, color tone
    # Returns a text description injected into the LLM prompt
```

This runs entirely on CPU using Pillow — no GPU required for image feature extraction.

---

### 5. Agent Tools (`src/agents/tools.py`)

The agent has access to four vision tools registered as LangChain tools:

| Tool | Description |
|------|-------------|
| `image_captioning` | Generate a detailed caption for an image |
| `visual_grounding` | Locate objects by returning bounding box coordinates |
| `object_detection` | Detect all objects with confidence scores |
| `vqa` | Answer a natural language question about an image |
| `image_to_base64` | Encode image as base64 for API payloads |

These tools are designed to be swapped with real model backends (e.g., DETR for detection, BLIP-2 for VQA).

---

### 6. Training Pipeline (`src/training/`)

**Pretraining (`pretrain.py`):**
- Contrastive loss (CLIP-style) aligns image and text embeddings
- Generative loss (cross-entropy) trains the LLM to generate captions given visual tokens
- Mixed loss: `total_loss = contrastive_loss + generative_loss`
- AdamW optimizer with linear warmup scheduler

**DPO Fine-tuning (`dpo.py`):**
- Direct Preference Optimization trains the model to prefer human-preferred responses over rejected ones
- No reward model needed — uses reference model log-probabilities directly
- Beta coefficient controls KL divergence penalty from reference model

---

### 7. Evaluation (`src/evaluation/evaluate.py`)

Evaluation uses two frameworks:

- **RAGAS** — Measures answer faithfulness, context relevance, and answer relevancy for RAG-style pipelines
- **DeepEval** — Tests hallucination, coherence, and task-specific metrics
- Adversarial testing and fairness checks are configurable in `config.yaml`

---

## Setup & Installation

### Prerequisites

- Windows 10/11 (or Linux/macOS)
- Python 3.11
- NVIDIA GPU (optional — CPU-only works)
- [Ollama](https://ollama.ai) installed and running

### Step 1 — Clone the repository

```bash
git clone https://github.com/trinath122/vision-language-agent.git
cd vision-language-agent
```

### Step 2 — Create virtual environment

```powershell
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

### Step 3 — Install PyTorch with CUDA

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```powershell
pip install torch torchvision
```

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 5 — Pull the LLM via Ollama

```powershell
ollama pull deepseek-r1:1.5b
```

### Step 6 — Set environment variables (optional — moves model cache off C: drive)

```powershell
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\model_cache\huggingface", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\ollama\models", "User")
```

---

## Running the Agent

### Text-only query

```powershell
python main.py --query "explain what a vision language agent does"
```

### Query with image

```powershell
python main.py --query "what do you see in this image?" --image "D:\naruto.png"
```

### With custom iteration limit

```powershell
python main.py --query "describe the mood of this image" --image "D:\photo.jpg" --max-iter 5
```

### Example output

```
Query: what do you see in this image?
Image: D:\naruto.png

Analyzing image with CLIP...
CLIP detected: Image properties: 1586x1169 pixels, bright image,
dominant color is RGB(216,186,155) suggesting warm/red-orange tones.

Agent Response:
The image has a bright, warm red-orange dominant color (RGB: 216, 186, 155),
suggesting warmth and energy. The high resolution (1586x1169) indicates a
detailed illustration with rich visual content...
```

---

## Configuration

All settings are in `configs/config.yaml`:

```yaml
model:
  vision_encoder: "openai/clip-vit-large-patch14"   # CLIP model variant
  language_model: "distilgpt2"                        # HuggingFace LLM (fallback)
  embedding_dim: 512                                  # CLIP output dimension
  projection_dim: 768                                 # LLM input dimension

training:
  pretrain:
    batch_size: 32
    learning_rate: 1.0e-4
    epochs: 10
  dpo:
    beta: 0.1                  # KL penalty coefficient
    learning_rate: 5.0e-5

quantization:
  enabled: false
  load_in_8bit: false          # Enable for 7B+ models on limited VRAM

agent:
  max_iterations: 10           # Max reasoning steps before stopping
```

---

## Design Decisions & Tradeoffs

### Why LangGraph for the agent loop?

LangGraph provides a **stateful, graph-based execution model** which gives explicit control over:
- When to stop reasoning (conditional edges)
- When to call tools vs. generate a final response
- How to maintain conversation history across turns

Alternatives like simple LangChain chains are less flexible for multi-step reasoning with tool use.

### Why Ollama for LLM inference?

Ollama runs quantized models locally without requiring Python-level model loading. This avoids the memory-mapping issues that occur when loading large safetensors files on Windows with limited RAM. It also makes the system easier to deploy — just `ollama pull <model>`.

### Why PIL instead of CLIP for image analysis in the agent?

The GTX 1650 (4GB VRAM) cannot simultaneously hold both the CLIP model and the LLM in memory. PIL-based analysis uses ~0MB GPU memory and runs in milliseconds. The tradeoff is losing semantic understanding (object recognition, scene classification) in exchange for system stability. On hardware with 16GB+ VRAM, full CLIP zero-shot analysis can be re-enabled.

### Why deepseek-r1:1.5b?

It's the smallest model in the Ollama library that reliably handles instruction-following. At 1.1GB, it runs within the 8GB RAM constraint alongside the Python process. The `:1.5b` suffix refers to 1.5 billion parameters.

---

## Hardware Constraints & Adaptations

This project was developed and tested on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce GTX 1650 (4GB VRAM) |
| RAM | 8GB |
| OS | Windows 11 |
| CUDA | 13.0 |
| Python | 3.11.9 |

### Adaptations made for this hardware:

1. **Removed CLIP from agent loop** — 4GB VRAM cannot hold both CLIP (605MB) and deepseek simultaneously without memory fragmentation
2. **PIL replaces CLIP zero-shot** — Extracts color, brightness, and resolution metadata instead of semantic embeddings
3. **Ollama for LLM** — Avoids Windows safetensors memory-mapping issue (OSError 1455) that crashes Python when loading large `.safetensors` files
4. **OLLAMA_NUM_GPU=0** — Forces Ollama to use CPU so GPU is not double-allocated
5. **Virtual memory tuned** — Windows page file expanded to 16GB to support model loading

### On 16GB+ RAM / 8GB+ VRAM systems, the intended full pipeline would be:

```
Image → CLIP ViT-L/14 → VisionProjection → [llava or phi-3-vision] → Response
```

---

## Future Work

- **Swap PIL for CLIP zero-shot classification** once on hardware with 8GB+ VRAM
- **Integrate llava:latest** for native image understanding (requires 8GB VRAM)
- **Add real tool backends** — DETR for object detection, BLIP-2 for VQA
- **Build a Gradio/Streamlit web UI** for interactive use
- **Run DPO fine-tuning** on a curated image-question-answer preference dataset
- **Deploy to GCP Vertex AI** using the included `deployment/gcp_deploy.py`
- **Add RAGAS evaluation** on a vision QA benchmark dataset

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `transformers` | HuggingFace model loading |
| `open-clip-torch` | CLIP vision encoder |
| `langgraph` | Agent reasoning graph |
| `langchain` | Tool definitions and message types |
| `ollama` | Local LLM inference |
| `Pillow` | Image loading and analysis |
| `peft` | LoRA fine-tuning |
| `trl` | DPO training |
| `ragas`, `deepeval` | Evaluation metrics |
| `bitsandbytes` | INT8 quantization |

---

## License

MIT License. See `LICENSE` for details.
