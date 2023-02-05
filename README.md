# teller_of_tales
Create narrated video story from book chapter using NLP, OpenAI and StableDiffusion. 
*Video creation process is automatic and unsupervised, results may vary.

![alt text](https://github.com/dawmro/teller_of_tales/blob/main/screenshot.png?raw=true)

## Prerequisites:
1. Python 3.8.10
2. NVidia GPU with 8GB VRAM. 

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


## Usage:
1. Delete audio, images, texts, videos directiories.
2. Paste your story into story.txt file
3. Run python script:
``` sh
python .\teller_of_tales.py
```
