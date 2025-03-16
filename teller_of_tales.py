# if first run then run installer 
# import nltk
# nltk.download()

from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime
import time
import openai
import json
import os
import re

import torch

from moviepy.editor import *
from tqdm.auto import tqdm

import fnmatch
from pathlib import Path

import edge_tts
import asyncio

from threading import Thread

import multiprocessing
from multiprocessing import Process

# for extracting keywords
from keybert import KeyBERT

import pathlib
import configparser

from moviepy.editor import (VideoFileClip, AudioFileClip, CompositeAudioClip)
from moviepy.audio.fx.all import volumex
import moviepy.video.fx.all as vfx

import requests
import io
import base64
from PIL import Image, PngImagePlugin
import shutil

import psutil

from fake_useragent import UserAgent

from moviepy.config import change_settings
change_settings({"FFMPEG_BINARY":"ffmpeg"})

from ollama import chat
from ollama import ChatResponse

import gc


config_path = pathlib.Path(__file__).parent.absolute() / "config.ini"
#BG_MUSIC_PATH = pathlib.Path(__file__).parent.absolute() / "bg_music/Fantasy Music - Passing the Crown - Avery Alexander (youtube).mp3"
config = configparser.ConfigParser()
config.read(config_path)

SPLIT_TEXT_ONLY = config["GENERAL"]["SPLIT_TEXT_ONLY"]
DEBUG = config["GENERAL"]["DEBUG"]
SPEED_UP = config["GENERAL"]["SPEED_UP"]
FREE_SWAP = int(config["GENERAL"]["FREE_SWAP"])
FPS = int(config["GENERAL"]["FPS"])

FRAGMENT_LENGTH = int(config["TEXT_FRAGMENT"]["FRAGMENT_LENGTH"])

USE_ELEVENLABS = config["AUDIO"]["USE_ELEVENLABS"]
ELEVENLABS_VOICE_ID = config["AUDIO"]["ELEVENLABS_VOICE_ID"]
USING_F5_TTS = config["AUDIO"]["USING_F5_TTS"]
VOICE = config["AUDIO"]["VOICE"]
BG_MUSIC = config["AUDIO"]["BG_MUSIC"]
BG_MUSIC_PATH = pathlib.Path(__file__).parent.absolute() / config["AUDIO"]["BG_MUSIC_PATH"]
MUSIC_VOLUME = float(config["AUDIO"]["MUSIC_VOLUME"])

USE_CHATGPT = config["IMAGE_PROMPT"]["USE_CHATGPT"]
model_engine = config["IMAGE_PROMPT"]["model_engine"]
OLLAMA_MODEL = config["IMAGE_PROMPT"]["OLLAMA_MODEL"]

seed = int(config["STABLE_DIFFUSION"]["seed"])
image_width = int(config["STABLE_DIFFUSION"]["image_width"])
image_height = int(config["STABLE_DIFFUSION"]["image_height"])
possitive_prompt_prefix = config["STABLE_DIFFUSION"]["possitive_prompt_prefix"]
possitive_prompt_sufix = config["STABLE_DIFFUSION"]["possitive_prompt_sufix"]
negative_prompt = config["STABLE_DIFFUSION"]["negative_prompt"]

USE_SD_VIA_API = config["STABLE_DIFFUSION"]["USE_SD_VIA_API"]
SD_URL = config["STABLE_DIFFUSION"]["SD_URL"]

USE_CHARACTERS_DESCRIPTIONS = config["STABLE_DIFFUSION"]["USE_CHARACTERS_DESCRIPTIONS"]
# descriptions should help maintain a consistent appearance of generated characters
if USE_CHARACTERS_DESCRIPTIONS == 'yes':
    CHARACTERS_DESCRIPTIONS = None
    char_desc_path = pathlib.Path(__file__).parent.absolute() / "characters_descriptions.ini"
  
    if os.path.exists(char_desc_path):
        char_desc = configparser.ConfigParser()
        char_desc.read(char_desc_path)
        if not char_desc["CHARACTERS_DESCRIPTIONS"]:
            CHARACTERS_DESCRIPTIONS = None
        else:
            CHARACTERS_DESCRIPTIONS = dict(char_desc["CHARACTERS_DESCRIPTIONS"])
   
    
if USE_ELEVENLABS == 'yes':
    # Use ELEVENLABS_API_KEY imported from environment variables
    ELEVENLABS_API_KEY = os.environ['ELEVENLABS_API_KEY']

if USE_CHATGPT == 'yes':
    # Use API_KEY imported from environment variables
    openai.api_key = os.environ['OPENAI_TOKEN']
      
                
def write_file(file_content, filename):
    if DEBUG == 'yes':
        print("Started writing file_content data into a file")
    with open(filename, "w", encoding='utf-8') as fp:
        fp.write(file_content)
        if DEBUG == 'yes':
            print("Done file_content data into a file")
        
        
def read_file(filename):
    with open(filename, "r", encoding='utf-8') as fp:
        file_content = fp.read()
        return file_content
    

def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    
def createFolders():
    
    if not os.path.exists("text"):
        os.makedirs("text") 
    if not os.path.exists("text/story_sentences"):
        os.makedirs("text/story_sentences") 
    if not os.path.exists("text/story_fragments"):
        os.makedirs("text/story_fragments") 
    if not os.path.exists("text/image_prompts"):
        os.makedirs("text/image_prompts") 
    if not os.path.exists("audio"):
        os.makedirs("audio")         
    if not os.path.exists("images"):
        os.makedirs("images")     
    if not os.path.exists("videos"):
        os.makedirs("videos")


def clean_text(text: str) -> str:
    """
    Clean the input text by replacing special characters and formatting.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    # Define a dictionary of replacements
    replacements = {
        'é': 'e',
        '>':'',
        '<':'',
        '=':'',
        '#':'',
        '..': '.',
        '“': '',
        '”': '',
        '-': ' ',
        '–': ' ',
        '—': ' ',
        '*':'',
        '_': '',
        '~':'',
        'XXXXXX':'',
        'xxxxx':'',
        '.....': '.',
        '....': '.',
        '...': ', ',
        '…': ', ',
        '\n\n\n': '\n',
        '\n\n': '\n'
    }

    # Sort the replacement keys by length in descending order
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)

    # Apply the replacements
    for key, value in sorted_replacements:
        text = text.replace(key, value)

    return text


def load_and_split_to_sentences(filename: str) -> int:
    """
    Load a story from a file, clean and split it into sentences, and write each sentence to a separate file.

    Args:
        filename (str): The path to the file containing the story.

    Returns:
        int: The number of sentences in the story.
    """
    # read raw story from txt file
    with open(filename, "r", encoding="utf-8") as file:
        story_raw = file.read()
        
    # Clean the input text by replacing special characters and formatting    
    story = clean_text(story_raw)

    # split story into list of sentences
    story_sentences_list = sent_tokenize(story)

    # split long sentence into multiple sentences
    new_story_sentences_list = []
    frag_len = 3*FRAGMENT_LENGTH
    punctuation_list = [',', ';', ':']
    for sentence in story_sentences_list:
        words = sentence.split()
        if len(words) <= frag_len:
            new_story_sentences_list.append(sentence)
        else:
            new_sentence = []
            for word in words:
                new_sentence.append(word)
                if word[-1] in punctuation_list and len(new_sentence) > frag_len:
                    new_story_sentences_list.append(' '.join(new_sentence))
                    new_sentence = []
            if new_sentence:
                new_story_sentences_list.append(' '.join(new_sentence))

    for i, story_sentence in enumerate(new_story_sentences_list):
        write_file(story_sentence, f"text/story_sentences/story_sentence{i}.txt")
    
    if DEBUG == 'yes':
        # display story enumerating through each sentence
        for i, story_sentence in enumerate(new_story_sentences_list):
            print( i, story_sentence)
        print("\n!!!!!!!!!!!!!!\nThis is last chance to make changes in story_sentences.txt files\n!!!!!!!!!!!!!!")
        pause()
    
    dir_path = r'text/story_sentences'
    number_of_files = len(fnmatch.filter(os.listdir(dir_path), 'story_sentence*.txt'))
    print('number_of_files:', number_of_files)
    
    return number_of_files



def sentences_to_fragments(number_of_story_sentences, FRAGMENT_LENGTH):

    # story divided into fragments
    story_fragments = []

    # fragment currently being worked on
    current_fragment = None

    # current fragment word counter
    current_fragment_word_counter = 0

    # for every sentence in list of sentences
    # combine sentences form story into fragments
    for i in range(number_of_story_sentences):
        
        # load current sentence
        story_sentence = read_file(f"text/story_sentences/story_sentence{i}.txt")
        
        # insert story sentence if current fragment is empty
        if current_fragment == None:
            current_fragment = story_sentence   
            
        # add story sentence to current fragment    
        else:
            current_fragment += ' ' + story_sentence
            
        # get amount of words in fragment    
        current_fragment_word_counter = len(word_tokenize(current_fragment))
        
        # if minimal length requirement is meet
        if current_fragment_word_counter > FRAGMENT_LENGTH:
            if DEBUG == 'yes':
                print(current_fragment_word_counter)
        
            # add current fragment to story fragments
            story_fragments.append(current_fragment)
            
            # zero temporary variables
            current_fragment = None
            current_fragment_word_counter = 0
     
    # add last fragment 
    if current_fragment is not None:
        story_fragments.append(current_fragment)
    
    for i, story_fragment in enumerate(story_fragments):
        write_file(story_fragment, f"text/story_fragments/story_fragment{i}.txt")
    
    if DEBUG == 'yes':
        # display story enumerating through each sentence
        for i, story_fragment in enumerate(story_fragments):
            print( i, story_fragment)
        print("\n!!!!!!!!!!!!!!\nThis is last chance to make changes in story_fragments.txt files\n!!!!!!!!!!!!!!")
        pause()
        
    dir_path = r'text/story_fragments'
    number_of_files = len(fnmatch.filter(os.listdir(dir_path), 'story_fragment*.txt'))
    print('number_of_files:', number_of_files)
    
    return number_of_files
    
    
    
def check_for_characters(story_fragment, char_desc_dict):
    matched_chars = []
    for name, desc in char_desc_dict.items():
        pattern = r'(?i)\b(' + name + r')\b'
        match = re.search(pattern, story_fragment)
        if match:
            matched_chars.append((name, desc))
            break  # Stop the loop after finding the first match
            
    result = ''
    if matched_chars:
        result = ', '.join([desc for _, desc in matched_chars])
        result = "[[[ " + result + " ]]]"
        result += ', '
        
    return result




def workaround_when_chatbot_refuses_to_answer(image_prompt: str, story_fragment: str):
    ngram_range = (1, 8)
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(
        story_fragment,
        keyphrase_ngram_range=ngram_range, 
        stop_words='english', 
        highlight=False,
        top_n=1
    )
    keywords_list = list(dict(keywords).keys())
    del kw_model
    del keywords
    # Force a garbage collection
    gc.collect()
    image_prompt = ', '.join(keywords_list)
    return image_prompt




def fragment_toPrompt(i, CURRENT_PROJECT_DIR, image_width: int=1, image_height: int=1, speedup: bool=False):
    story_fragment = read_file(f"{CURRENT_PROJECT_DIR}/text/story_fragments/story_fragment{i}.txt")
    print(f"{i} Fragment: {story_fragment}")
    
    # probably needs some improvements
    prefix = f"You are an expert in crafting intricate prompts for the generative AI 'Stable Diffusion XL'. Create a prompt that does not conflict with the following style and setting of the story: {possitive_prompt_sufix}. Respond only with image prompt and nothing else. Suggest good image prompt to illustrate the following fragment from story, make description illustrative, precise and detailed, one sentence, max 20 words: "
    
    if USE_CHATGPT == 'yes':
       
        # translate fragment into prompt
        image_prompt = askChatGPT(prefix + story_fragment, model_engine).strip()
        
    elif USE_CHATGPT == 'ollama':
        
        # translate fragment into prompt 
        response: ChatResponse = chat(model=OLLAMA_MODEL, messages=[
            {
                'role': 'user',
                'content': prefix + story_fragment,
            },
        ])
        image_prompt = (response['message']['content']).replace("“", '').replace("”", '').replace("‘", "'").replace("’", "'").replace('"', '')
            
    else:
        image_prompt = workaround_when_chatbot_refuses_to_answer(image_prompt,story_fragment)
    
    # I cannot provide information on how to create explicit content. 
    # Can I help you with something else?
    if any(phrase in image_prompt for phrase in ["I can", "?"]):
        print("Chatbot refuses to answer, using KeyBERT instead")
        image_prompt = workaround_when_chatbot_refuses_to_answer(image_prompt,story_fragment)
        
    if USE_CHARACTERS_DESCRIPTIONS == 'yes':    
        if CHARACTERS_DESCRIPTIONS != None:
            image_prompt = f"{check_for_characters(story_fragment, CHARACTERS_DESCRIPTIONS)}{image_prompt}"
           
    print(f"{i} Created Prompt: {image_prompt}")
    write_file(image_prompt, f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt") 
    
    # vvv if using pollinations generate image immediately when Prompt is ready
    if (True if isinstance(USE_SD_VIA_API, str) and USE_SD_VIA_API == "pollinations" else False) and (speedup == True) and (i%5 == 0):
        print(f"{showTime()} {i} Frag_to_prompt_thread: Starting immediate prompt_to_image ...")
        pollinations_thread = Process(target=prompt_to_image, args=(i, image_width, image_height, CURRENT_PROJECT_DIR, True, True))
        pollinations_thread.start() 
    # ^^^ if using pollinations generate image immediately when Prompt is ready
      
      
    
def prompt_to_image(i, image_width, image_height, CURRENT_PROJECT_DIR, try_once: bool=False, wait: bool=False):
    do_it = True
    wait_time = 10
    image_prompt = read_file(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt")
    print(f"{i} Loaded Prompt: {image_prompt}")
    while(do_it):
        try:
            if USE_SD_VIA_API == 'yes':
                url = SD_URL

                payload = {
                    "prompt": f"{possitive_prompt_prefix} {image_prompt} {possitive_prompt_sufix}",
                    "negative_prompt": f"{negative_prompt}",
                    #"steps": 10,
                    "steps": 22,
                    "width": image_width,
                    "height": image_height,
                    "height": image_height,
                    "seed": -1,
                    #"guidance_scale": "1.6",
                    "guidance_scale": "4.0",
                    #"sampler_index": "DPM++ 2M SDE Karras",
                    #"sampler_index": "DPM++ 3M SDE Exponential",
                    #"sampler_index": "DPM++ 2M Karras",
                    "sampler_index": "Euler a",
                    #"sampler_index": "DPM++ 3M SDE Karras",
                    #"sampler_index": "DPM++ SDE",
                    
                }
                
                option_payload = {
                    #"sd_model_checkpoint": "animagineXLV31_v31.safetensors",
                    #"sd_model_checkpoint": "dreamshaperXL10_alpha2Xl10.safetensors [0f1b80cfe8]",
                    #"sd_model_checkpoint": "animeArtDiffusionXL_alpha3.safetensors",
                    "sd_model_checkpoint": "aamXLAnimeMix_v10.safetensors",
                    #"sd_model_checkpoint": "sdxlUnstableDiffusers_v11Rundiffusion.safetensors",
                    #"sd_model_checkpoint": "lomoxl_.safetensors",
                    #"sd_model_checkpoint": "sdxlYamersAnime_stageAnima.safetensors",
                    "sd_vae": "sdxl_vae.safetensors",
                }
                
                requests.post(url=f"{url}/sdapi/v1/options", json=option_payload)
                response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

                r = response.json()
                
                for j in r['images']:
                    image = Image.open(io.BytesIO(base64.b64decode(j.split(",",1)[0])))

                    png_payload = {
                        "image": "data:image/png;base64," + j
                    }
                    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

                    pnginfo = PngImagePlugin.PngInfo()
                    pnginfo.add_text("parameters", response2.json().get("info"))
                    image.save(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg", pnginfo=pnginfo)
                    
            elif USE_SD_VIA_API == 'pollinations':
                if wait:
                    time.sleep(i)    
                
                prompt = f"{possitive_prompt_prefix} {image_prompt} {possitive_prompt_sufix}"
                #url = f"https://image.pollinations.ai/prompt/{prompt}?width={image_width}&height={image_height}&model=flux&nologo=true&enhance=true&seed={time.time()}"
                url = f"https://image.pollinations.ai/prompt/{prompt}?width={image_width}&height={image_height}&nologo=true&model=flux&enhance=falsee&seed={time.time()}&negative=nsfw"
                
                HEADERS = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "Cache-Control": "max-age=0",
                }
                
                ua = UserAgent()
                HEADERS["User-Agent"] = ua.random
 
                response = requests.get(url=url, headers=HEADERS, timeout=60)
                print(f"{showTime()} {i}: {response.text[:100]}")
                if response.status_code == 200:
                    image = io.BytesIO(response.content)
                    img = Image.open(image)
                    if(Path(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").is_file() == False):
                        img.save(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg")
                else:
                    raise requests.exceptions.HTTPError(f'Failed to download the image. Status code: {response.status_code}')    
                
            else:
                pass
                
            do_it = False
            
        except Exception as e:
            if try_once:
                do_it = False
                print(f"Exception!!! {i} \n{e} \nNot trying again.")
            else:    
                print(f"Exception!!! {i} \n{e} \nWaiting for {wait_time} seconds and trying again...")
                time.sleep(wait_time)


async def create_vioceover(story_fragment, CURRENT_PROJECT_DIR) -> None:
    
    TEXT = story_fragment
    OUTPUT_FILE = f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav"
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)
        
        
def create_elevenlabs_vioceover(story_fragment, CURRENT_PROJECT_DIR) -> None:
    
    # check subscription status
    url = "https://api.elevenlabs.io/v1/user/subscription"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    response = requests.request("GET", url, headers=headers)
    char_count = json.loads(response.text)['character_count']
    char_limit = json.loads(response.text)['character_limit']
    if (char_limit - char_count) < 500:
        raise ValueError(f"{char_limit - char_count} Characters remaining, renew your Elevenlabs subscription!")
    
    else:
        OUTPUT_FILE_MP3 = f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.mp3"
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

        headers = {
          "Accept": "audio/mpeg",
          "Content-Type": "application/json",
          "xi-api-key": ELEVENLABS_API_KEY
        }
        
        data = {
          "text": story_fragment,
          "model_id": "eleven_multilingual_v2",
          "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
          }
        }

        response = requests.post(url, json=data, headers=headers)
        with open(OUTPUT_FILE_MP3, 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    
                    
                    
def create_kokoro_voiceover(i, story_fragment, CURRENT_PROJECT_DIR) -> None:
    
    story_fragment=story_fragment.replace("\n", " ")
    Type="wav"
    OUTPUT_FILE_MP3 = f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.{Type}"
    CHUNK_SIZE = 1024
    
    url = f"http://localhost:8880/v1/audio/speech"
    json={
            "model": "kokoro",
            "input": story_fragment,
            "voice": "af_heart+af_nicole",
            "speed": 1.1,
            "response_format": Type,
            "stream": True,
        }

    response = requests.post(url=url, json=json, stream=True)
    with open(OUTPUT_FILE_MP3, 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)                    
    
    
   
def createVideoClip(i, CURRENT_PROJECT_DIR):

    story_fragment = read_file(f"{CURRENT_PROJECT_DIR}/text/story_fragments/story_fragment{i}.txt")

    # load the audio file using moviepy
    try:
        audio_clip = AudioFileClip(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.mp3")
    except:
        audio_clip = AudioFileClip(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav")
    
    # FIX: 
    # ffmepg incorrectly reporting the duration of the audio.
    # This causes iter_chunks in AudioClip to try to read frames outside the length of the file. 
    # In FFMPEG_AudioReader get_frame reads the end of the file again, resulting in a glitch.
    # Cut 0.05 from the end to remove glitch
    # Note: using ffmpeg 6.0 instead of default 4.2.2 from imageio_ffmpeg could be possible fix, 
    # it also reduces metalic noise in audio.
    if USING_F5_TTS == "yes":
        audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.5)
    else:
        audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.1)
    
    # add audio fadein / fadeout ot minimize sound glitches
    audio_clip = audio_clip.audio_fadein(0.05).audio_fadeout(0.05)
    
    silence_duration = 0.5
    if USE_ELEVENLABS != 'no':
        silence_duration = 0.7
        
    silence = AudioClip(make_frame = lambda t: 0, duration = silence_duration)
    # add 0.5 second silence to begining of audio 
    audio_clip = concatenate_audioclips([silence, audio_clip])
    # add 0.5 second silence to end of audio
    audio_clip = concatenate_audioclips([audio_clip, silence])
    
    # get audio duration
    audio_duration = audio_clip.duration
    
    # use short video clips instead of images (making final video trully animated)
    # example: animate generated imageX.jpg files using pikalabs and save them in images
    # directory as movieX.mp4 files
    if(Path(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").is_file() == True):
        movie_clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").loop(duration = audio_duration)
        
    elif(Path(f"{CURRENT_PROJECT_DIR}/images/movie{i}.mp4").is_file() == True):
        # load movie fragment file using moviepy
        movie_clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/images/movie{i}.mp4")
        reversed_movie_clip = movie_clip.fx(vfx.time_mirror)
        mirrored_movie_clip = concatenate_videoclips([movie_clip, reversed_movie_clip], padding=-0.2, method="compose")
        movie_clip = mirrored_movie_clip.resize( (image_width, image_height) )
        movie_clip.write_videofile(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4", fps=FPS, codec="h264_nvenc")
        movie_clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").loop(duration = audio_duration)

    else:
        # load the image file using moviepy
        image_clip = ImageClip(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").set_duration(audio_duration)
    
    # use moviepy to create a text clip from the text
    screensize = (image_width, image_height)
    text_clip = TextClip(story_fragment, fontsize=int(0.0599*image_height), font="Impact", color="black", stroke_color="white", stroke_width=round(0.0026*image_height, 1), size=screensize, method='caption', align="South")
    text_clip = text_clip.set_duration(audio_duration)
    
    # concatenate the audio, image, and text clips
    if(Path(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").is_file() == True):
        clip = movie_clip.set_audio(audio_clip)
    else:
        clip = image_clip.set_audio(audio_clip)
    video = CompositeVideoClip([clip, text_clip])
    
    # save Video Clip to a file
    video_mp4 = video.write_videofile(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4", fps=FPS, codec="h264_nvenc")
    print(f"{showTime()} The Video{i} Has Been Created Successfully!")
        
    
def askChatGPT(text, model_engine):
    do_it = True
    answer = ''
    while(do_it):
        try: 
            completions = openai.Completion.create(
                engine=model_engine,
                prompt=text,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.9,
                request_timeout=10.0, # test
            )
            do_it = False
            answer = completions.choices[0].text
        except Exception as e:
            wait_time = 10
            print(f"Exception!!! \n{e} \nWaiting for {wait_time} seconds and trying again...")
            time.sleep(wait_time)
    
    return answer  
    
 
 
def createListOfClips(CURRENT_PROJECT_DIR):
    
    clips = []
    l_files = os.listdir(CURRENT_PROJECT_DIR+"/videos")
    l_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    #no_digit_files = [f for f in l_files if not re.search('\d', f)]
    #l_files.sort(key=lambda f: int(re.sub('\D', '', f)) if re.search('\d', f) else float('inf'))
    #print("Files without digits:")
    #for file in no_digit_files:
    #    print(file)
    
    for file in l_files:
        clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/videos/{file}")
        clips.append(clip)
    
    return clips


    

def makeFinalVideo(project_name, CURRENT_PROJECT_DIR):

    # create sorted list of clips
    print(f"{showTime()} Fixing order of video clips")
    clips = createListOfClips(CURRENT_PROJECT_DIR)
    
    # add audio fade to prevent audio glitches when combining multiple clips
    print(f"{showTime()} Adding audio fadein / fadeout...")
    clips = [clip.audio_fadein(0.10).audio_fadeout(0.10) for clip in clips]
    
    # add video fade to create smooth transitions
    print(f"{showTime()} Adding video fadein / faedout...")
    clips = [clip.crossfadein(1.0).crossfadeout(1.0) for clip in clips]
    
    # combine all clips into final video
    print(f"{showTime()} Concatenate all clips into final video...")
    final_video = concatenate_videoclips(clips, padding=-1, method="compose")
    
    # add backgroud music to video
    if BG_MUSIC == "yes":
        print(f"{showTime()} Adding music to file...")
        original_audio = final_video.audio
        soundtrack = AudioFileClip(str(BG_MUSIC_PATH))
        bg_music = soundtrack.audio_loop(duration=final_video.duration)
        bg_music = bg_music.volumex(MUSIC_VOLUME)
        final_audio = CompositeAudioClip([original_audio, bg_music])
        final_video = final_video.set_audio(final_audio)
    
    print(f"{showTime()} Writing final video to file...")
    final_video.write_videofile(CURRENT_PROJECT_DIR+'/'+project_name+".mp4", fps=FPS, codec="h264_nvenc")
        
    print(f"{showTime()} Final video created successfully!")

    
    

if __name__ == "__main__":
    
    print(f"{showTime()}")
    # Get current working directory
    CWD = os.getcwd()
    # set name for main directory for projects
    PROJECTS_DIR = 'projects'
    # list each project in PROJECTS_DIR
    project_names_mixed = [ f.name for f in os.scandir(PROJECTS_DIR) if f.is_dir() ]
    
    # sort project directiories by name
    project_names = []
    try:
        project_names_mixed.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    except:
        pass
    for project_name_mixed in project_names_mixed:
        project_names.append(project_name_mixed)
       
    # run each project in PROJECTS_DIR in sequence
    for project_name in project_names:
        processes_list = []
        CURRENT_PROJECT_DIR = CWD+'/'+PROJECTS_DIR+'/'+project_name
        os.chdir(CURRENT_PROJECT_DIR)
        print("Current working directory: {0}".format(os.getcwd())) 
        
        if(Path(f"{CURRENT_PROJECT_DIR}/{project_name}.mp4").is_file() == False):
            if DEBUG == 'yes':
                pause()
                
            # Create directiories for text, audio, images and video files    
            createFolders()
            
            if len(os.listdir(f"{CURRENT_PROJECT_DIR}/text/story_fragments")) == 0:
                # load story and split it by sentence
                number_of_story_sentences = load_and_split_to_sentences("story.txt")
                
                # group sentences into story fragments of a given length
                number_of_story_fragments = sentences_to_fragments(number_of_story_sentences, FRAGMENT_LENGTH)
                
            else:
                number_of_story_fragments = len(os.listdir(f"{CURRENT_PROJECT_DIR}/text/story_fragments"))
            
            if SPLIT_TEXT_ONLY == "yes":
                continue
            
            image_prompts = []
            
            using_video_fragments_Processes = False
  
  
            # vvv ollama pre-generate 
            if USE_CHATGPT == 'ollama':
                response = requests.post(url=f"{SD_URL}/sdapi/v1/unload-checkpoint", json={})
                print(response.text)
                time.sleep(1)
                for o in range(number_of_story_fragments):
                    if(Path(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{o}.txt").is_file() == False):
                        # translate fragment into prompt
                        fragment_toPrompt(o, CURRENT_PROJECT_DIR, image_width, image_height, True if isinstance(SPEED_UP, str) and SPEED_UP == "yes" else False)
                        
                # unload ollama model        
                url = 'http://localhost:11434/api/generate'
                data = {'model': OLLAMA_MODEL, 'keep_alive': 0}
                response = requests.post(url, json=data)
                print(response.text)
                time.sleep(1)
                response = requests.post(url=f"{SD_URL}/sdapi/v1/reload-checkpoint", json={})
                print(response.text)
                time.sleep(1)
            # ^^^ ollama pre-generate 
                
                
            # for each story fragment
            for i in range(number_of_story_fragments):
                print(f"{showTime()} {i} of {number_of_story_fragments-1}:")
                
                # vvvvvv pause / unpause
                # if cpu usage is more than 95%, wait (tweak this value based on your needs) 
                cpu_usage = int(psutil.cpu_percent(interval=0.1, percpu=False))
                while (cpu_usage > 90):
                    print(f"{showTime()} Main: High CPU usage! {cpu_usage}% -> Waiting...")
                    time.sleep(2)
                    cpu_usage = int(psutil.cpu_percent(interval=2.0, percpu=False))
                # ^^^^^^ pause / unpause
                
                #vvv
                # if free virtual memory is less than this amount of GB, then wait (tweak this value based on your needs)
                free_swap = int(psutil.swap_memory()[2]/1000000000)
                while (free_swap < FREE_SWAP):
                    print(f"{showTime()} Main: High vmem usage! Free swap: {free_swap} -> Waiting...")
                    time.sleep(300)
                    free_swap = int(psutil.swap_memory()[2]/1000000000)
                #^^^

                # create voiceover 
                if(Path(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav").is_file() == False) and (Path(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.mp3").is_file() == False):
                    story_fragment = read_file(f"text/story_fragments/story_fragment{i}.txt")
                    do_it = True
                    while(do_it):
                        try: 
                            if USE_ELEVENLABS == 'elevenlabs':
                                create_elevenlabs_vioceover(story_fragment, CURRENT_PROJECT_DIR)
                                print(f"{showTime()} Created voiceover using Elevenlabs...")
                            elif USE_ELEVENLABS == 'kokoro':
                                create_kokoro_voiceover(i, story_fragment, CURRENT_PROJECT_DIR)
                                print(f"{showTime()} Created voiceover using Kokoro...")
                            else:
                                asyncio.get_event_loop().run_until_complete(create_vioceover(story_fragment, CURRENT_PROJECT_DIR))
                                print(f"{showTime()} Created voiceover using Edge-tts...")
                            do_it = False
                        except Exception as e:
                            wait_time = 10
                            print(f"Exception!!! \n{e} \nWaiting for {wait_time} seconds and trying again...")
                            time.sleep(wait_time)
                    
                if(Path(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt").is_file() == False):
                    # translate fragment into prompt
                    fragment_toPrompt(i, CURRENT_PROJECT_DIR, image_width, image_height, False)
                
                        
                if(Path(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").is_file() == True) and (Path(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4").is_file() == False):
                    # do not start all image to video conversions at once
                    time.sleep(2)
                
                if(Path(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").is_file() == False):
                    # generate image form prompt
                    prompt_to_image(i, image_width, image_height, CURRENT_PROJECT_DIR)
                    
                if(Path(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4").is_file() == False):
                    # create video clip using story fragment and generated image
                    # create a new process
                    using_video_fragments_Processes = True
                    process = Process(target = createVideoClip, args = (i, CURRENT_PROJECT_DIR))
                    # start the new process
                    process.start()
                    processes_list.append(process)
 
            if(using_video_fragments_Processes):
                # wait for the new process to finish
                print(f"{showTime()} Main: Waiting for video_fragments process to terminate...")
                # block until all tasks are done
                for process in processes_list:
                    process.join() # call to ensure subsequent line (e.g. restart_program) 
                    # is not called until all processes finish
                time.sleep(10)
                # continue on
                print(f"{showTime()} Main: video_fragments joined, continuing on")

            
            if(Path(CURRENT_PROJECT_DIR+'/'+project_name+".mp4").is_file() == False):
                # create final video
                final_video_process = Process(target = makeFinalVideo, args = (project_name, CURRENT_PROJECT_DIR))
                final_video_process.start()
                time_to_wait = int(((i/10)+1)*FPS)
                print(f"{showTime()} Waiting {time_to_wait} second before starting next project")
                time.sleep(time_to_wait)

                # block until all tasks are done    
                #final_video_process.join()
                #print('Main: final_video_process joined, continuing on')
                #pause()

        
   
   
'''
# voices for edge-tts:

Name: en-AU-NatashaNeural
Gender: Female

Name: en-AU-WilliamNeural
Gender: Male

Name: en-CA-ClaraNeural
Gender: Female

Name: en-CA-LiamNeural
Gender: Male

Name: en-GB-LibbyNeural
Gender: Female

Name: en-GB-MaisieNeural
Gender: Female

Name: en-GB-RyanNeural
Gender: Male

Name: en-GB-SoniaNeural
Gender: Female

Name: en-GB-ThomasNeural
Gender: Male

Name: en-HK-SamNeural
Gender: Male

Name: en-HK-YanNeural
Gender: Female

Name: en-IE-ConnorNeural
Gender: Male

Name: en-IE-EmilyNeural
Gender: Female

Name: en-IN-NeerjaExpressiveNeural
Gender: Female

Name: en-IN-NeerjaNeural
Gender: Female

Name: en-IN-PrabhatNeural
Gender: Male

Name: en-KE-AsiliaNeural
Gender: Female

Name: en-KE-ChilembaNeural
Gender: Male

Name: en-NG-AbeoNeural
Gender: Male

Name: en-NG-EzinneNeural
Gender: Female

Name: en-NZ-MitchellNeural
Gender: Male

Name: en-NZ-MollyNeural
Gender: Female

Name: en-PH-JamesNeural
Gender: Male

Name: en-PH-RosaNeural
Gender: Female

Name: en-SG-LunaNeural
Gender: Female

Name: en-SG-WayneNeural
Gender: Male

Name: en-TZ-ElimuNeural
Gender: Male

Name: en-TZ-ImaniNeural
Gender: Female

Name: en-US-AnaNeural
Gender: Female

Name: en-US-AriaNeural
Gender: Female

Name: en-US-ChristopherNeural
Gender: Male

Name: en-US-EricNeural
Gender: Male

Name: en-US-GuyNeural
Gender: Male

Name: en-US-JennyNeural
Gender: Female

Name: en-US-MichelleNeural
Gender: Female

Name: en-US-RogerNeural
Gender: Male

# good one
Name: en-US-SteffanNeural
Gender: Male

Name: en-ZA-LeahNeural
Gender: Female

Name: en-ZA-LukeNeural
Gender: Male
'''