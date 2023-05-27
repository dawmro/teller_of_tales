# teller_of_tales

## Features:
* Create narrated video story from book chapter using NLP, OpenAI and StableDiffusion.
* You can use keyBERT instead of ChatGPT as a free option. 
* You can run multiple projects at once to create multiple narrated videos. 
* Video creation process is automatic and unsupervised, results may vary.
Example: https://youtu.be/fz-Ez8PsE5A

![alt text](https://github.com/dawmro/teller_of_tales/blob/main/screenshot.png?raw=true)

## How does it work (simplified workflow):
1. User places chapter of a book into 'story.txt' file in 'projects/my_project_name' directory.
2. User starts program with 'python .\teller_of_tales.py' command.
3. User goes away to do his own things.
4. Text in chapter is split into separate sentences.
5. Sentences are combined into fragments of appropriate length.
6. Each text fragment is converted to audio file using text to speach.
7. For each fragment, prompts are created using ChatGPT or KeyBERT. Prompts describe situation that 'can be seen' in a given fragment.
8. Using StableDiffusion and prompts from point 7. image is created.
9. Audio and image files are transformed into video file of current scene using MoviePy.
10. Video files of separate scenes are concatenated into final video file.
11. User comes back and watches final video file.

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
6a. Don't want to use OpenAI account? No problem! Make sure that USE_CHATGPT in line 37 is set to False:
``` sh
USE_CHATGPT = False
```
7. Login to HugginFace using your Access Token from https://huggingface.co/settings/tokens:
``` sh
huggingface-cli login
```



## Usage:
1. Create folder in 'projects' directory. Folder name will become final video name.
2. Paste your story into story.txt file inside created folder.
3. Create multiple folders and paste multiple stories if you want to run multiple projects at once.
4. Run python script:
``` sh
python .\teller_of_tales.py
```
