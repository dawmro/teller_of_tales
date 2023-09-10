# teller_of_tales

Teller of Tales is a project that creates narrated video stories from book chapters using natural language processing (NLP), OpenAI, and StableDiffusion. It can run multiple projects at once and generate videos automatically and unsupervised. The results may vary depending on the input text and the chosen options.

## Features:
* NLP with OpenAI or KeyBERT
* Image generation with StableDiffusion
* Text to speech with Edge Text-to-Speech
* Video editing with MoviePy

Here is an example of a video story created by Teller of Tales: https://youtu.be/fz-Ez8PsE5A

![alt text](https://github.com/dawmro/teller_of_tales/blob/main/screenshot.png?raw=true)

## How does it work:
1. User places a chapter of a book into the ‘story.txt’ file in the ‘projects/my_project_name’ directory.
2. User starts the program with the ‘python .\teller_of_tales.py’ command.
3. User goes away to do something else.
4. Text in the chapter is split into separate sentences.
5. Sentences are combined into fragments of appropriate length.
6. Each text fragment is converted to an audio file using text to speech.
7. For each fragment, prompts are created using ChatGPT or KeyBERT. Prompts describe the situation that 'can be seen' in a given fragment.
8. Using StableDiffusion and prompts from point 7, image is created.
9. Audio and image files are transformed into a video file of the current scene using MoviePy.
10. Video files of separate scenes are concatenated into the final video file.
11. User comes back and watches the final video file.

## Prerequisites:
1. Python 3.8.10
2. NVidia GPU with 4GB VRAM. 

## Setup:
1. Create new virtual env:
``` sh
python -m venv env
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



## Usage:
1. Create a folder in the ‘projects’ directory. The folder name will become the final video name.
2. Paste your story into the story.txt file inside the created folder.
3. Create multiple folders and paste multiple stories if you want to run multiple projects at once.
4. Run the python script:
``` sh
python .\teller_of_tales.py
```
5. Wait for the script to finish and check the folder with profect for the output video.
