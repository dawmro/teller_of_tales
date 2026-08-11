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


For each story, the application:

1. Reads `story.txt`.
2. Cleans and splits the text into sentences.
3. Groups sentences into story fragments.
4. Generates an image prompt for each fragment.
5. Generates narration for each fragment.
6. Generates an image for each fragment.
7. Combines the image, narration, and story text into a video clip.
8. Concatenates the clips into a final MP4 video.
9. Optionally adds background music.

The application is project-oriented: each story is stored in its own directory under `projects/`.

---

## How it works

The processing pipeline is:

```text
projects/<project_name>/story.txt
                │
                ▼
         Text preprocessing
                │
                ▼
          Sentence splitting
                │
                ▼
          Story fragments
                │
                ▼
        Image prompt generation
                │
        ┌───────┴────────┐
        ▼                ▼
   TTS narration     Image generation
        │                │
        └───────┬────────┘
                ▼
         Video clip creation
                │
                ▼
       Individual MP4 clips
                │
                ▼
        Final video assembly
                │
                ▼
projects/<project_name>/<project_name>.mp4
```

Image prompt generation, narration generation, and image generation are performed as separate sequential stages.

Video clips are generated in parallel using up to 8 worker processes.

---



## Features

* Convert plain-text stories into narrated videos.
* Automatically split stories into manageable fragments.
* Generate image prompts with:

  * Ollama
  * OpenAI's legacy completion API
  * KeyBERT
* Generate images using:

  * Stable Diffusion WebUI / Automatic1111 API
  * Pollinations
* Generate narration using:

  * Edge TTS
  * ElevenLabs
  * Kokoro
* Add the story fragment as text over the generated image.
* Add background music to the final video.
* Generate video clips in parallel.
* Keep intermediate files for inspection and reuse.
* Skip already-generated intermediate files.
* Optionally add configured character descriptions to image prompts.
* Continue processing a project without regenerating existing assets.

---
  
![alt text](https://github.com/dawmro/teller_of_tales/blob/main/docs/screenshot.png?raw=true)

## Example video output
https://github.com/user-attachments/assets/079fa221-9822-45d7-be65-bdc7b5f2db38

---

## Project structure

Create one directory for each story:

```text
projects/
└── my_story/
    └── story.txt
```

The application creates the required output directories automatically.

After processing, the project will contain files similar to:

```text
projects/
└── my_story/
    ├── story.txt
    │
    ├── text/
    │   ├── story_sentences/
    │   │   ├── story_sentence0.txt
    │   │   ├── story_sentence1.txt
    │   │   └── ...
    │   │
    │   ├── story_fragments/
    │   │   ├── story_fragment0.txt
    │   │   ├── story_fragment1.txt
    │   │   └── ...
    │   │
    │   └── image_prompts/
    │       ├── image_prompt0.txt
    │       ├── image_prompt1.txt
    │       └── ...
    │
    ├── audio/
    │   ├── voiceover0.wav
    │   ├── voiceover1.wav
    │   └── ...
    │
    ├── images/
    │   ├── image0.jpg
    │   ├── image1.jpg
    │   └── ...
    │
    ├── videos/
    │   ├── video0.mp4
    │   ├── video1.mp4
    │   └── ...
    │
    └── my_story.mp4
```

The exact audio extension depends on the selected TTS provider.

---

# Text processing

## Input

The input file must be:

```text
projects/<project_name>/story.txt
```

It is read as UTF-8 text.

## Text cleaning

Before sentence splitting, the script normalizes several characters and sequences.

Among other transformations, it:

* Removes selected special characters.
* Replaces hyphens and different dash characters with spaces.
* Removes underscores and asterisks.
* Normalizes some repeated punctuation.
* Converts `...` and `…` to comma-like separators.
* Removes some placeholder strings.
* Reduces repeated blank lines.

The original `story.txt` is not overwritten.

## Sentence splitting

The cleaned text is passed to NLTK's `sent_tokenize()`.

Sentences longer than `FRAGMENT_LENGTH` are additionally split at commas, semicolons, and colons once the accumulated part is longer than approximately three times `FRAGMENT_LENGTH`.

The resulting sentences are written to:

```text
text/story_sentences/story_sentence<N>.txt
```

## Story fragments

The generated sentences are then grouped consecutively.

Sentences are accumulated until the current group contains more than `FRAGMENT_LENGTH` words.

The resulting fragments are written to:

```text
text/story_fragments/story_fragment<N>.txt
```

The final fragment may contain fewer words than `FRAGMENT_LENGTH`.

---

# Image prompt generation

An image prompt is generated for every story fragment.

The prompt-generation instruction asks the provider to create a single-sentence Stable Diffusion XL scene description of no more than 20 words.

The configured style suffix is included in the instruction.

Three modes are supported by the code.

## Ollama

Set:

```ini
[IMAGE_PROMPT]
IMAGE_PROMPT_PROVIDER = ollama
OLLAMA_MODEL = <ollama-model>
```

The application sends the fragment to the configured Ollama model.

If the Ollama request fails, the application falls back to KeyBERT.

## OpenAI

Set:

```ini
[IMAGE_PROMPT]
IMAGE_PROMPT_PROVIDER = chatgpt
```

The current implementation uses:

```text
text-davinci-003
```

through the legacy OpenAI Completion API.

The API key is read from:

```text
OPENAI_TOKEN
```

If the OpenAI request fails, the application falls back to KeyBERT.

> This is a legacy OpenAI completion integration. It should not be confused with the current OpenAI chat-model APIs.

## KeyBERT

Any provider value other than `chatgpt` or `ollama` causes the application to use the KeyBERT fallback directly.

KeyBERT uses:

```text
all-mpnet-base-v2
```

and extracts one keyphrase using an n-gram range of 1–8 words.

KeyBERT is also used automatically if an LLM provider fails or returns a response containing an apparent refusal or question mark.

---

# Character descriptions

Character descriptions are optional.

Enable them with:

```ini
[STABLE_DIFFUSION]
USE_CHARACTERS_DESCRIPTIONS = yes
```

When enabled, the application reads:

```text
characters_descriptions.ini
```

If the file exists, character descriptions are loaded from the `CHARACTERS_DESCRIPTIONS` section.

For each fragment, the script searches for configured character names.

If a matching character is found, its description is prepended to the generated image prompt.

The current implementation uses the first matching configured character.

Character descriptions are therefore a simple prompt enhancement mechanism; they do not provide guaranteed character consistency between generated images.

---

# Image generation

The generated prompt is expanded with the configured positive prompt prefix and suffix:

```text
<positive prefix> <generated prompt> <positive suffix>
```

A negative prompt is also configured for Stable Diffusion.

Two image-generation backends are supported.

## Stable Diffusion WebUI / Automatic1111

Set:

```ini
[STABLE_DIFFUSION]
USE_SD_VIA_API = yes
SD_URL = http://127.0.0.1:7860
```

The application uses the Automatic1111 `txt2img` API.

The current request uses:

```text
Steps:          20
Sampler:        Euler a
CFG / guidance: 4.0
Seed:            configured value
Width:           configured value
Height:          configured value
```

The script also requests the following model configuration from the Stable Diffusion API:

```text
aamXLAnimeMix_v10.safetensors
sdxl_vae.safetensors
```

Therefore these models need to be available in the configured Stable Diffusion WebUI installation.

Generated images are saved as:

```text
images/image<N>.jpg
```

## Pollinations

Set:

```ini
[STABLE_DIFFUSION]
USE_SD_VIA_API = pollinations
```

The application sends the prompt to the Pollinations image API.

The request uses:

```text
model=flux
enhance=false
nologo=true
```

The configured image dimensions are passed to the API.

A time-based seed is used for each Pollinations request.

## Image generation retries

Image generation is retried indefinitely if an exception occurs.

The initial retry delay is 10 seconds.

---

# Text-to-speech

The narration is generated separately for every story fragment.

Three TTS providers are supported.

## Edge TTS

Any `TTS_PROVIDER` value other than `elevenlabs` or `kokoro` uses Edge TTS.

Example:

```ini
[AUDIO]
TTS_PROVIDER = edge
VOICE = en-US-BrianNeural
```

The narration is saved as:

```text
audio/voiceover<N>.wav
```

## Kokoro

Set:

```ini
[AUDIO]
TTS_PROVIDER = kokoro
KOKORO_URL = http://localhost:8880/v1/audio/speech
KOKORO_VOICE_ID = af_heart
```

The application sends the fragment text to the configured Kokoro HTTP endpoint.

The text is converted to lowercase before being sent.

The request asks for WAV output.

Generated audio is saved as:

```text
audio/voiceover<N>.wav
```

## ElevenLabs

Set:

```ini
[AUDIO]
TTS_PROVIDER = elevenlabs
ELEVENLABS_VOICE_ID = <voice-id>
```

The implementation uses the ElevenLabs multilingual v2 model.

The generated audio is saved as:

```text
audio/voiceover<N>.mp3
```

The ElevenLabs implementation also checks the account character usage before requesting the narration.

### Current ElevenLabs implementation note

The current `teller_of_tales.py` references `ELEVENLABS_API_KEY` inside `tts_elevenlabs()`, but does not define that variable before using it.

Therefore the ElevenLabs path currently requires a code fix before it can be considered fully functional.

---

# Video clip creation

Each story fragment becomes one video clip.

The clip combines:

* The generated image.
* The generated narration.
* The original story fragment as a text overlay.

## Audio processing

Before creating the clip:

1. The last 0.1 seconds of the generated audio are removed.
2. A 50 ms fade-in is applied.
3. A 50 ms fade-out is applied.
4. 0.5 seconds of silence are added before the narration.
5. 0.5 seconds of silence are added after the narration.

The resulting audio determines the duration of the video clip.

## Image

The generated image remains on screen for the entire duration of the processed audio.

## Text overlay

The original story fragment is rendered over the image.

The current implementation uses:

```text
Font:          Impact
Alignment:     South
Method:        caption
Text color:    black
Stroke color:  white
```

The font size and stroke width scale with the configured image height.

## Output

Each clip is written to:

```text
videos/video<N>.mp4
```

using:

```text
H.264 / libx264
```

and the configured FPS.

---

# Parallel video rendering

Image generation and TTS are performed sequentially.

Video clip creation is the only explicitly parallelized stage.

The application uses `ProcessPoolExecutor` with:

```text
max_workers = min(number_of_CPU_cores, 8)
```

Therefore no more than 8 worker processes are used for video clip creation.

A one-second delay is inserted between submitting clip-generation tasks.

---

# Final video

After all individual clips have been created, they are sorted numerically:

```text
video0.mp4
video1.mp4
video2.mp4
...
```

The clips receive:

* 1-second crossfade-in.
* 1-second crossfade-out.

They are then concatenated using MoviePy's `compose` method with one second of overlap.

The final video is written to:

```text
projects/<project_name>/<project_name>.mp4
```

using:

```text
libx264
```

at the configured FPS.

---

# Background music

Background music is optional.

Configuration:

```ini
[AUDIO]
BG_MUSIC = yes
BG_MUSIC_PATH = bg_music/example.mp3
MUSIC_VOLUME = 0.05
```

When enabled:

1. The background music is looped to the length of the final video.
2. Its volume is multiplied by `MUSIC_VOLUME`.
3. It is mixed with the video's existing audio.

The narration remains part of the final video's audio track.

---

# Configuration

The application reads configuration from:

```text
config.ini
```

## General

```ini
[GENERAL]

DEBUG = False
SPEED_UP = False
FREE_SWAP = 0
FPS = 10
```

### `DEBUG`

Enables timestamped diagnostic logging.

```ini
DEBUG = True
```

### `SPEED_UP`

The value is loaded by the script, but the current processing pipeline does not use it to change behavior.

### `FREE_SWAP`

The value is loaded by the script, but the current processing pipeline does not use it to control processing.

### `FPS`

Controls the frame rate used when rendering individual clips and the final video.

---

## Text fragments

```ini
[TEXT_FRAGMENT]

FRAGMENT_LENGTH = 10
```

`FRAGMENT_LENGTH` controls the approximate target size of generated story fragments in words.

It also influences how unusually long sentences are split.

---

## Audio

```ini
[AUDIO]

TTS_PROVIDER = edge

ELEVENLABS_VOICE_ID = <voice-id>

KOKORO_VOICE_ID = af_heart
KOKORO_URL = http://localhost:8880/v1/audio/speech

VOICE = en-US-BrianNeural

BG_MUSIC = no
BG_MUSIC_PATH = bg_music/music.mp3
MUSIC_VOLUME = 0.05
```

### `TTS_PROVIDER`

Supported behavior:

```text
elevenlabs  → ElevenLabs
kokoro      → Kokoro
anything else → Edge TTS
```

---

## Image prompt generation

```ini
[IMAGE_PROMPT]

IMAGE_PROMPT_PROVIDER = ollama
OLLAMA_MODEL = llama3.1:8b-instruct-q8_0
```

Supported providers:

```text
ollama
chatgpt
```

Any other value uses KeyBERT directly.

---

## Stable Diffusion

```ini
[STABLE_DIFFUSION]

positive_prompt_prefix = ...
positive_prompt_suffix = ...
negative_prompt = ...

USE_SD_VIA_API = yes
SD_URL = http://127.0.0.1:7860

seed = -1

image_width = 1344
image_height = 768

USE_CHARACTERS_DESCRIPTIONS = no
```

### `USE_SD_VIA_API`

Supported behavior:

```text
yes
    Use Stable Diffusion WebUI / Automatic1111.

pollinations
    Use Pollinations.

anything else
    No image-generation backend is selected.
```

The last option effectively does nothing when `generate_image()` is called, so a valid image backend should normally be selected.

### `seed`

The value is passed directly to the Stable Diffusion API.

Pollinations uses its own time-based seed regardless of this setting.

---

# Environment variables

Depending on the selected providers, the application expects API credentials from environment variables.

## OpenAI

When:

```ini
IMAGE_PROMPT_PROVIDER = chatgpt
```

the application reads:

```text
OPENAI_TOKEN
```

## ElevenLabs

The intended API credential is:

```text
ELEVENLABS_API_KEY
```

However, the current ElevenLabs implementation contains a variable-definition issue described above.

---

# External services

Depending on configuration, Teller of Tales can communicate with:

| Component                              | Purpose                                  |
| -------------------------------------- | ---------------------------------------- |
| Ollama                                 | Local image-prompt generation            |
| OpenAI                                 | Legacy LLM-based image-prompt generation |
| Stable Diffusion WebUI / Automatic1111 | Local image generation                   |
| Pollinations                           | Remote image generation                  |
| Edge TTS                               | Text-to-speech                           |
| ElevenLabs                             | Text-to-speech                           |
| Kokoro                                 | Local/API text-to-speech                 |

You only need the services corresponding to the providers you select.

---

# Dependencies

The Python application uses libraries including:

* `edge-tts`
* `openai`
* `psutil`
* `requests`
* `fake-useragent`
* `keybert`
* `moviepy`
* `nltk`
* `ollama`
* `Pillow`

The exact Python package versions should be installed from the project's `requirements.txt`.

The application also requires external tools/services depending on the selected configuration.

## FFmpeg

MoviePy is explicitly configured to use:

```text
ffmpeg
```

FFmpeg must therefore be installed and available in the system `PATH`.

Verify it with:

```bash
ffmpeg -version
```

## ImageMagick

MoviePy's `TextClip` implementation may require ImageMagick depending on the MoviePy version and environment.

The `Impact` font must also be available to the text-rendering environment.

## NLTK data

The application uses:

```python
nltk.tokenize.sent_tokenize
```

Therefore the required NLTK tokenizer data must be installed before processing a story.

---


## Prerequisites:
1. Python 3.8.10
2. NVidia GPU with 4GB VRAM. 

## Setup:

1. Clone the repository
``` sh
git clone https://github.com/dawmro/teller_of_tales.git
cd teller_of_tales
```

2. Create new virtual env:
``` sh
py -3.8 -m venv env
```

3. Activate your virtual env:
``` sh
env/Scripts/activate
```

4. Install PyTorch from https://pytorch.org/get-started/locally/:
``` sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

5. Install packages from included requirements.txt:
``` sh
pip install -r .\requirements.txt
```

6. Install ImageMagick:
``` sh
https://imagemagick.org/script/download.php

Add both check boxes:
* Associate supported file extensions
* Install legacy utilities
```

7. Install and configure the services you intend to use

For example:

* Ollama for local prompt generation.
* Stable Diffusion WebUI for local image generation.
* Kokoro for local TTS.

Alternatively, configure remote providers such as Pollinations, Edge TTS, OpenAI, or ElevenLabs where appropriate.

8. Configure the application

Edit:

```text
config.ini
```

Select the desired image-generation and TTS providers.

---



# Usage

From the repository root, create a project:

```text
projects/
└── my_story/
    └── story.txt
```

Place the story text in:

```text
projects/my_story/story.txt
```

Then run:

```bash
python teller_of_tales.py
```

The script scans:

```text
./projects/
```

and processes every directory found there.

Only directories containing:

```text
story.txt
```

are processed.

The final result is:

```text
projects/my_story/my_story.mp4
```

---

# Multiple stories

Multiple stories can be processed by placing them in separate project directories:

```text
projects/
├── story_one/
│   └── story.txt
│
├── story_two/
│   └── story.txt
│
└── story_three/
    └── story.txt
```

Running:

```bash
python teller_of_tales.py
```

processes the project directories in sorted order.

---

# Resuming an interrupted project

The application stores intermediate results on disk and checks whether they already exist before generating them again.

The following files can be reused:

```text
story fragments
image prompts
voiceovers
images
video clips
final video
```

For example, if:

```text
images/image5.jpg
```

already exists, that image is not generated again.

Likewise, an existing:

```text
audio/voiceover5.wav
```

or:

```text
audio/voiceover5.mp3
```

prevents the narration from being generated again.

Existing video clips are also skipped.

Finally, if the project already contains:

```text
<project_name>.mp4
```

the final video is not rendered again.

This allows a partially completed project to be restarted without automatically repeating completed stages.

---

# Processing stages in detail

For a project containing `N` story fragments, the script performs approximately:

```text
1 × story preprocessing
N × image prompts
N × voiceovers
N × images
N × video clips
1 × final video
```

The video clips are the only stage that is explicitly parallelized.

---

# Output files

## Sentence files

```text
text/story_sentences/story_sentence<N>.txt
```

Contain the cleaned and sentence-split story.

## Fragment files

```text
text/story_fragments/story_fragment<N>.txt
```

Contain the text used for one narrated video segment.

## Image prompts

```text
text/image_prompts/image_prompt<N>.txt
```

Contain the generated image prompt before the configured positive prefix and suffix are added.

## Narration

```text
audio/voiceover<N>.wav
```

or:

```text
audio/voiceover<N>.mp3
```

depending on the TTS provider.

## Images

```text
images/image<N>.jpg
```

## Video clips

```text
videos/video<N>.mp4
```

## Final video

```text
<project_name>.mp4
```

---

# Resource management

The script contains functions for unloading and reloading Ollama and Stable Diffusion checkpoints.

Before image-prompt generation, it attempts to reload Ollama.

Before image generation, it attempts to reload Stable Diffusion.

The purpose is to manage GPU memory when both an LLM and an image-generation model are used on the same machine.

These operations only execute for the corresponding provider:

```text
Ollama
Stable Diffusion WebUI
```

They do not affect Pollinations or other providers.

---

# Limitations and current implementation notes

This README documents the behavior of the current `teller_of_tales.py`. Several implementation details are worth knowing.

## Legacy OpenAI API

The OpenAI prompt-generation implementation uses:

```text
text-davinci-003
```

through the legacy Completion API.

A modern OpenAI API integration may therefore require code changes.

## ElevenLabs variable issue

The current ElevenLabs function references `ELEVENLABS_API_KEY`, but that variable is not defined by the script.

The ElevenLabs path should be fixed before relying on it.

## No automatic model downloading

The script does not download Stable Diffusion, Ollama, Kokoro, or other large models automatically.

Those services and their models must be installed separately.

## Image generation is not parallelized

Image generation happens one image at a time.

Only video clip creation uses multiple processes.

## TTS is not parallelized

Narration generation also happens one fragment at a time.

## Character consistency is not guaranteed

Character descriptions are inserted into prompts when a configured character name is found.

The system does not use reference images, embeddings, LoRAs, or another dedicated character-consistency mechanism.

## Failed image generation retries indefinitely

An image-generation exception causes the script to wait and retry.

There is no maximum retry count in the current implementation.

## Existing fragments are treated as authoritative

If `text/story_fragments/` already contains fragment files, the script does not regenerate the sentence/fragment structure from `story.txt`.

To regenerate the text segmentation after changing `story.txt` or `FRAGMENT_LENGTH`, remove the existing generated fragment files/directories before running the application again.

## Existing final video is not automatically rebuilt

If the final `<project_name>.mp4` already exists, the final rendering stage is skipped.

Delete the final MP4 when a new final render is required.

---

# Typical workflow

A typical local setup can look like:

```text
Story
  │
  ▼
story.txt
  │
  ▼
Teller of Tales
  │
  ├── NLTK
  │     └── sentence splitting
  │
  ├── Ollama
  │     └── image prompts
  │
  ├── Kokoro
  │     └── narration
  │
  ├── Stable Diffusion WebUI
  │     └── illustrations
  │
  └── MoviePy + FFmpeg
        └── video clips + final MP4
```

An internet-based configuration can instead use:

```text
Story
  │
  ▼
Teller of Tales
  │
  ├── OpenAI / KeyBERT
  │     └── image prompts
  │
  ├── Edge TTS / ElevenLabs
  │     └── narration
  │
  ├── Pollinations
  │     └── illustrations
  │
  └── MoviePy + FFmpeg
        └── final MP4
```

---

# Repository structure

The main executable is:

```text
teller_of_tales.py
```

The main configuration files are:

```text
config.ini
characters_descriptions.ini
```

Stories and generated assets are stored under:

```text
projects/
```

The application does not require the notebook to run.

---
