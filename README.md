[![ChatGPT](https://img.shields.io/badge/ChatGPT-OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-FF6B35?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48dGV4dCB4PSI1MCIgeT0iNjAiIGZvbnQtc2l6ZT0iODAiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+TzwvdGV4dD48L3N2Zz4=)](https://ollama.ai/)
[![KeyBERT](https://img.shields.io/badge/KeyBERT-NLP-3776AB?logo=python&logoColor=white)](https://github.com/MaartenGr/KeyBERT)
[![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-2C3E50?logo=python&logoColor=white)](https://www.nltk.org/)
[![Python 3.10](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Edge-TTS](https://img.shields.io/badge/Edge%20TTS-Microsoft-0078D4?logo=microsoft&logoColor=white)](https://github.com/rany2/edge-tts)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-Voice%20AI-00D9FF?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48dGV4dCB4PSI1MCIgeT0iNjAiIGZvbnQtc2l6ZT0iOTAiIGZvbnQtd2VpZ2h0PSJib2xkIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+🔊</L3RleHQ+PC9zdmc+)](https://elevenlabs.io/)
[![MoviePy](https://img.shields.io/badge/MoviePy-Video%20Editing-E34C26?logo=movie&logoColor=white)](https://zulko.github.io/moviepy/)
[![StableDiffusion](https://img.shields.io/badge/Stable%20Diffusion-Image%20Gen-512BD4?logo=pytorch&logoColor=white)](https://stablediffusion.ai/)


# Teller of Tales

Transform your favorite book chapters into stunning narrated video stories with AI-powered automation.

## Overview

**Teller of Tales** is an intelligent automation system that converts written book chapters into professional-quality narrated videos. By leveraging natural language processing, advanced language models, and AI image generation, the project creates compelling visual narratives with synchronized voiceovers, background music, and text overlays—all in a fully automated, scalable pipeline.

This tool enables content creators, educational platforms, and storytellers to generate engaging video content at scale, reducing production time from hours to minutes while maintaining high quality output.


## Features:
* NLP with OpenAI, Ollama or KeyBERT
* Image generation with StableDiffusion
* Text to speech with Edge Text-to-Speech or Elevenlabs
* Video editing with MoviePy
  
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/screenshot.png?raw=true)

## Example video output
https://github.com/user-attachments/assets/079fa221-9822-45d7-be65-bdc7b5f2db38



## Components & Flow
### 1. File Input
- Input File: projects/[project_name]/story.txt
- Action: User provides a text file containing a chapter.
- Output: Folder structure initialized for project.

### 2. Text Preprocessing & Splitting
Components:
- Components involved in splitting text:
- Text Storage → Text Splitter → Sentence Fragmentator

- Text Storage: Loads story.txt using read_file().
- Text Cleaner: Uses clean_text() to normalize text (remove special chars).
- Sentence Splitter: Uses NLTK sent_tokenize() to split into sentences.
- Fragment Aggregator: Combines sentences into ~N-word fragments (FRAGMENT_LENGTH) for manageable processing.

### 3. Concurrent Processing Pipeline
The following steps run in parallel per fragment (managing CPU/memory via process pools):

```
sequenceDiagram
    User->>TextFragment: Process fragment
    TextFragment->>TTS: Generate audio
    TextFragment->>PromptEngine: Create prompt
    TextFragment->>ImageGen: Generate image
    loop Per fragment
        TTS->>AudioFile: Save WAV/MP3
        PromptEngine->>PromptFile: Save prompt text
        ImageGen->>ImageFile: Save JPG
    end
```
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/processing_pipeline.PNG?raw=true)

A. Text-to-Speech (TTS)
- Engines:
  - Edge TTS: Async via edge_tts.Communicate (default)
  - ElevenLabs: Synthesizes via API if configured
- Process:
  - Audio generated for each fragment.
  - Saves as audio/voiceover{i}.mp3 or .wav.

B. Prompt Generation
Strategies:
- LLM-Based:
  - ChatGPT: Asks "Craft a visual prompt from this scene".
  - Ollama: Offline LLM for prompt generation.
- KeyBERT (fallback):
  - Keyword extraction (NLTK + KeyBERT) if LLM fails.
- Output: Saved to text/image-prompts/image_prompt{i}.txt.

C. Stable Diffusion Image Generation
- Backends:
  - Local API (e.g., SD WebUI): Sends prompts to SD_URL.
  - Pollinations: Cloud API with requests (faster but less control).
- Process:
  - 1. Uses prompt file + global style desc.
  - 2. Saves image as images/image{i}.jpg.

### 4. Video Clips Creation
MoviePy Workflow (per fragment):

```
graph LR
  subgraph VideoClipProcess{i}
  Image --> ImageClip
  Audio --> AudioClip
  subgraph Compositing
    ImageClip --> [Background]
    TextClip --> [Foreground]
  end
  Compositing --> VideoClip
  end
```
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/video_clip_creation.PNG?raw=true)

- Audio Processing:
  - Crossfades
  - Silence padding
- Text Overlay:
  - Captions on image/movie clips.
- Output: videos/video{i}.mp4.

### 5. Final Video Assembly
Steps:

1. Clip Sorter: Orders video*.mp4 numerically.
2. Transition Layer:
- Crossfades/soft cuts between clips.
- Background music layering.
3. Encoder:
- H264 via moviepy.write_videofile.

### Dependency Graph

```
graph TD
  A[story.txt] --> B[Preprocessing]
  B --> |Sentences| C{Fragment Split}

  C --> |Frag#1| D[TTS → Audio]
  C --> |Frag#1| E[LLM → Prompt]
  E --> H1["image_prompt{i}.txt"]
  H1 --> F[Stable Diffusion → Image]
  F --> I1["image{i}.jpg"]

  D --> G1["voiceover{i}.wav"]
  
  G1 & I1 --> J[MoviePy Clip]
  J --> K["video{i}.mp4"]

  subgraph Aggregation
    K --> L[Final.mp4]
    style Aggregation fill:#f9f
  end

  L --> M[User Watch]
  style D fill:#f88,stroke:#cc0
  style E fill:#d8d
  style F fill:#a93
```
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/dependency_graph.PNG?raw=true)

### Concurrency Model
- Processing Mode:
  - Fragment jobs run in parallel (via multiprocessing).
  - IO-bound tasks (TTS, API calls) use async/threads.
- Resource Limits:
  - Checks CPU, memory, and swap (uses psutil).

### Usage Flow

```
graph LR
  StartUserInput[Place story.txt] --> StartScript[python teller.py]
  StartScript --> LoadProject[Project folder setup]
  LoadProject --> ProcessText[Split and fragment]
  ProcessText --> TTSPipeline[TTS Processing]
  ProcessText --> PromptGen[LLM Prompts]
  TTSPipeline --> AudioFiles
  PromptGen --> Prompts
  Prompts --> ImageGen[Images via SD]
  
  subgraph PerFragmentSteps["Per-fragment steps"]
    AudioFiles --> ClipAssembly[Audio+Image→Video]
    ImageGen --> ClipAssembly
    ClipAssembly --> VideoFragments
  end
```
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/usage_flow.PNG?raw=true)

This architecture balances parallelism while preventing system overload, leveraging modern APIs and affordable cloud services where needed.

## 🔧 Configuration Guide

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `FRAGMENT_LENGTH` | Int | 150 | Words per text fragment (lower = shorter clips, higher = longer) |
| `FREE_SWAP` | Int | 200 | Minimum GB free space before processing |
| `DEBUG` | Bool | no | Verbose logging and debug info |
| `USE_CHATGPT` | Bool | yes | Enable ChatGPT-based prompt generation |
| `USE_ELEVENLABS` | Bool | no | Enable premium ElevenLabs TTS |
| `SD_URL` | URL | localhost:7860 | StableDiffusion API endpoint |
| `OLLAMA_MODEL` | String | llama2:7b | Ollama model specification |
| `STYLE_DESCRIPTION` | String | empty | Global art style to apply to all images |

### Key Configuration Points
```
# config.ini snippet
[GENERAL]
FREE_SWAP=200  # GB free RAM for swapping
DEBUG=no

[AUDIO]
USE_ELEVENLABS=no # Or edge-tts

[IMAGE_PROMPT]
OLLAMA_MODEL=llama3.2:3b-instruct-q8_0 # Offline model path

[STABLE_DIFFUSION]
SD_URL=http://localhost:7860 # Local API URL
```

## Prerequisites:
1. Python 3.8.10
2. NVidia GPU with 4GB VRAM. 

## Setup:
1. Create new virtual env:
``` sh
py -3.8 -m venv env
```
2. Activate your virtual env:
``` sh
env/Scripts/activate
```
3. Install PyTorch from https://pytorch.org/get-started/locally/:
``` sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
4. Install packages from included requirements.txt:
``` sh
pip install -r .\requirements.txt
```
5. Install ImageMagick:
``` sh
https://imagemagick.org/script/download.php

Add both check boxes:
* Associate supported file extensions
* Install legacy utilities
```
6. Add your OpenAI Token from https://beta.openai.com/account/api-keys to environment variables:
``` sh
setx OPENAI_TOKEN=your_token
```
6a. Don't want to use OpenAI account? No problem! Make sure that USE_CHATGPT in config.ini is set to no:
``` sh
USE_CHATGPT = no
```
7. Login to HugginFace using your Access Token from https://huggingface.co/settings/tokens:
``` sh
huggingface-cli login
```



## 📖 Usage

### Basic Workflow

#### 1. Prepare Your Content

```
projects/
├── my_first_story/
│   └── story.txt          (Your book chapter goes here)
├── another_story/
│   └── story.txt
└── third_story/
    └── story.txt
```

Each `story.txt` file can be as long an you want (longer texts take proportionally longer to process).

#### 2. Run the Pipeline

```bash
python teller_of_tales.py
```

The script automatically discovers all projects in the `projects/` directory and processes them sequentially or in parallel based on your configuration.

#### 3. Retrieve Output

Each project folder contains the complete processing pipeline output:

```
projects/my_first_story/
├── story.txt                    # Original input
├── audio/
│   ├── voiceover0.mp3
│   ├── voiceover1.mp3
│   └── ...
├── images/
│   ├── image0.jpg
│   ├── image1.jpg
│   └── ...
├── text/
│   └── image-prompts/
│       ├── image_prompt0.txt
│       ├── image_prompt1.txt
│       └── ...
├── videos/
│   ├── video0.mp4
│   ├── video1.mp4
│   └── ...
└── final.mp4                    # 🎬 Your completed video!
```
