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

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

from gtts import gTTS
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

import psutil


config_path = pathlib.Path(__file__).parent.absolute() / "config.ini"
bg_music_path = pathlib.Path(__file__).parent.absolute() / "bg_music/bg_music2.mp3"

config = configparser.ConfigParser()
config.read(config_path)

DEBUG = config["GENERAL"]["DEBUG"]
SPEED_UP = config["GENERAL"]["SPEED_UP"]
FRAGMENT_LENGTH = int(config["TEXT_FRAGMENT"]["FRAGMENT_LENGTH"])

VOICE = config["AUDIO"]["VOICE"]
BG_MUSIC = config["AUDIO"]["BG_MUSIC"]

USE_CHATGPT = config["IMAGE_PROMPT"]["USE_CHATGPT"]
model_engine = config["IMAGE_PROMPT"]["model_engine"]

NSFW_filter = config["STABLE_DIFFUSION"]["NSFW_filter"]
lowmem = config["STABLE_DIFFUSION"]["lowmem"]
seed = int(config["STABLE_DIFFUSION"]["seed"])
image_width = int(config["STABLE_DIFFUSION"]["image_width"])
image_height = int(config["STABLE_DIFFUSION"]["image_height"])
model_id = config["STABLE_DIFFUSION"]["model_id"]
possitive_prompt_prefix = config["STABLE_DIFFUSION"]["possitive_prompt_prefix"]
possitive_prompt_sufix = config["STABLE_DIFFUSION"]["possitive_prompt_sufix"]
negative_prompt = config["STABLE_DIFFUSION"]["negative_prompt"]

USE_SD_VIA_API = config["STABLE_DIFFUSION"]["USE_SD_VIA_API"]


if USE_CHATGPT == 'yes':
    # Use API_KEY imported from environment variables
    openai.api_key = os.environ['OPENAI_TOKEN']


def write_list(a_list, filename):
    if DEBUG == 'yes':
        print("Started writing list data into a json file")
    with open(filename, "w") as fp:
        json.dump(a_list, fp)
        if DEBUG == 'yes':
            print("Done writing JSON data into .json file")


def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list
        
                
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


def load_and_split_to_sentences(filename):

    # read raw story from txt file
    with open(filename, "r", encoding="utf-8") as file:
        story_raw = file.read()

    # remove quotes from story
    # gate wotw, ares game, america stranded
    story = story_raw.replace('“', '').replace('”', '').replace('-', ' ').replace('—', ' ').replace('*', ' ').replace('(1)', '').replace('(2)', '').replace('(3)', '').replace('(4)', '').replace('(5)', '').replace('(6)', '').replace('(7)', '').replace('(8)', '').replace('(9)', '').replace('_', '').replace('.....', '').replace('....', '').replace('...', ', ').replace('~', ' ').replace('*', ' ')#.replace('Xxx', ' ').replace('xxx', ' ').replace('X x X', ' ').replace('X x x', ' ').replace('x x x', ' ').replace('X X X', ' ').replace('X', ' ').replace('\n\n', '\n')
    
    # summoning america, wait is this just gate, age of memeoris, america in another world
    #story = story_raw.replace('“', '').replace('”', '').replace('—', ' ').replace('    ', ' ')
    
    # lucius
    #story = story_raw.replace('“', '').replace('”', '').replace('—', ' ').replace('    ', ' ').replace(':', '.').replace(';', '.')
    
    # war of worlds wells
    #story = story_raw.replace('“', '').replace('”', '').replace('—', ' ').replace('*', ' ').replace('(1)', '').replace('(2)', '').replace('(3)', '').replace('(4)', '').replace('(5)', '').replace('(6)', '').replace('(7)', '').replace('(8)', '').replace('(9)', '').replace('_', '').replace('.....', '').replace('....', '').replace('...', '').replace('\n', ' ').replace('      ', ' ')

    # split story into list of sentences
    story_sentences_list = sent_tokenize(story)
    
    # write_list(story_sentences_list, "text/story_sentences_list.json")
    
    for i, story_sentence in enumerate(story_sentences_list):
        write_file(story_sentence, f"text/story_sentences/story_sentence{i}.txt")
    
    if DEBUG == 'yes':
        # display story enumerating through each sentence
        for i, story_sentence in enumerate(story_sentences_list):
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
    
    # write_list(story_fragments, "text/story_fragments.json")
    
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
      
    
def prompt_to_image(pipe, generator, i, image_width, image_height, CURRENT_PROJECT_DIR):
    do_it = True
    wait_time = 10
    image_prompt = read_file(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt")
    print(i, image_prompt)
    while(do_it):
        try:
            if USE_SD_VIA_API == 'no':
                if lowmem == "yes":
                    # restart StableDiffusionPipeline
                    pipe, generator = prepare_pipeline()
                
                # scale number of steps with image size to prevent large grainy images
                steps = int(min(((20 * image_height * image_width) / 407040) + 1, 50)) 
                
                prompt = possitive_prompt_prefix + image_prompt + possitive_prompt_sufix
                    
                image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=image_height, width=image_width, guidance_scale=6.0, generator=generator, num_inference_steps=steps).images[0]

                image.save(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg")  
            
            else:
                url = "http://127.0.0.1:7860"

                payload = {
                    "prompt": f"{possitive_prompt_prefix} {image_prompt} {possitive_prompt_sufix}",
                    "negative_prompt": f"{negative_prompt}",
                    "steps": 20,
                    "width": image_width,
                    "height": image_height,
                    "seed": -1,
                    "guidance_scale": "7.0",
                    "sampler": "DPM++ 2S a Karras",
                    "sd_model_checkpoint": "dreamshaperXL10_alpha2Xl10.safetensors [0f1b80cfe8]",
                    "sd_vae": "sdxl_vae.safetensors",
                }
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
                    
            do_it = False
            
        except Exception as e:
            print(f"Exception!!! \n{e} \nWaiting for {wait_time} seconds and trying again...")
            time.sleep(wait_time)


async def create_vioceover(story_fragment, CURRENT_PROJECT_DIR) -> None:
    
    TEXT = story_fragment
    OUTPUT_FILE = f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav"
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)
    
   
def createVideoClip(i, CURRENT_PROJECT_DIR):

    story_fragment = read_file(f"{CURRENT_PROJECT_DIR}/text/story_fragments/story_fragment{i}.txt")

    # load the audio file using moviepy
    audio_clip = AudioFileClip(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav")

    # FIX:
    # ffmepg incorrectly reporting the duration of the audio.
    # This causes iter_chunks in AudioClip to try to read frames outside the length of the file.
    # In FFMPEG_AudioReader get_frame reads the end of the file again, resulting in a glitch.
    # Cut 0.05 from the end to remove glitch
    # Note: using ffmpeg 6.0 instead of default 4.2.2 from imageio_ffmpeg could be possible fix,
    # it also reduces metalic noise in audio.
    audio_clip = audio_clip.subclip(0, audio_clip.duration - 0.05)
    
    # add audio fadein / fadeout ot minimize sound glitches
    audio_clip = audio_clip.audio_fadein(0.05).audio_fadeout(0.05)
    
    # add 1 second silence to begining of audio 
    silence = AudioClip(make_frame = lambda t: 0, duration = 1.0)
    audio_clip = concatenate_audioclips([silence, audio_clip])
    
    # get audio duration
    audio_duration = audio_clip.duration
    
    # use short video clips instead of images (making final video trully animated)
    # example: animate generated image_x.jpg files using pikalabs and save them in images
    # directory as movie_x.mp4 files
    if(Path(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").is_file() == True):
        movie_clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4").loop(duration = audio_duration)
        
    elif(Path(f"{CURRENT_PROJECT_DIR}/images/movie{i}.mp4").is_file() == True):
        # load movie fragment file using moviepy
        movie_clip = VideoFileClip(f"{CURRENT_PROJECT_DIR}/images/movie{i}.mp4")
        reversed_movie_clip = movie_clip.fx(vfx.time_mirror)
        mirrored_movie_clip = concatenate_videoclips([movie_clip, reversed_movie_clip], padding=-0.2, method="compose")
        movie_clip = mirrored_movie_clip.resize( (image_width, image_height) )
        movie_clip.write_videofile(f"{CURRENT_PROJECT_DIR}/images/movie_mirror{i}.mp4", fps=24)
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
    video_mp4 = video.write_videofile(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4", fps=24)
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



def fragment_toPrompt(i, CURRENT_PROJECT_DIR):
    story_fragment = read_file(f"{CURRENT_PROJECT_DIR}/text/story_fragments/story_fragment{i}.txt")
    print(f"{i} Fragment: {story_fragment}")
    
    if USE_CHATGPT == 'yes':
        prefix = "Suggest good image to illustrate the following fragment from story, make descrpition short and precise, one sentence, max 10 words: "
        # translate fragment into prompt
        image_prompt = askChatGPT(prefix + story_fragment, model_engine).strip()
        
    else:
        ngram_range = (1, 3)
        if USE_SD_VIA_API == 'yes':
            ngram_range = (1, 6)
        kw_model = KeyBERT(model='all-mpnet-base-v2')
        keywords = kw_model.extract_keywords(
            story_fragment,
            keyphrase_ngram_range=ngram_range, 
            stop_words='english', 
            highlight=False,
            top_n=2
        )
        keywords_list = list(dict(keywords).keys())
        
        image_prompt = ', '.join(keywords_list)       
    
    print(f"{i} Prompt: {image_prompt}")
    write_file(image_prompt, f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt")     
    

def createListOfClips(CURRENT_PROJECT_DIR):

    clips = []
    l_files = os.listdir(CURRENT_PROJECT_DIR+"/videos")
    l_files.sort(key=lambda f: int(re.sub('\D', '', f)))
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
        video_clip = final_video
        original_audio = video_clip.audio
        soundtrack = AudioFileClip(str(bg_music_path))
        bg_music = soundtrack.audio_loop(duration=video_clip.duration)
        bg_music = bg_music.volumex(0.04)
        final_audio = CompositeAudioClip([original_audio, bg_music])
        final_clip = video_clip.set_audio(final_audio)
        final_clip.write_videofile(CURRENT_PROJECT_DIR+'/'+project_name+".mp4", fps=24)
    else:
        final_video = final_video.write_videofile(CURRENT_PROJECT_DIR+'/'+project_name+".mp4", fps=24)
     
    print(f"{showTime()} Final video created successfully!")

    
def prepare_pipeline():
    # vvvvvvvvv prepare StableDiffusionPipeline 
    # clear cuda cache
    with torch.no_grad():
        torch.cuda.empty_cache()
        
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # if limited by GPU memory (4GB VRAM):
    if lowmem == 'yes': 
        # 1. do not move the pipeline to CUDA beforehand or else the gain in memory consumption will only be minimal
        # pipe = pipe.to("cuda")
        # 2. offload the weights to CPU and only load them to GPU when performing the forward pass
        pipe.enable_sequential_cpu_offload()
        # 3. consider chunking the attention computation  
        pipe.enable_attention_slicing(1)           
    else:
        pipe = pipe.to("cuda")
    
    # randomize seed
    if seed == -1:    
        generator = torch.Generator("cuda")
    # use manual seed    
    else:
        generator = torch.Generator("cuda").manual_seed(seed)
        
    def dummy_checker(images, **kwargs): return images, False
    if NSFW_filter == 'no':
        pipe.safety_checker = dummy_checker 
   
    return pipe, generator
    # ^^^^^^^^ prepare StableDiffusionPipeline 
    

if __name__ == "__main__":
    
    if USE_SD_VIA_API == 'no':
        # prepare StableDiffusionPipeline
        pipe, generator = prepare_pipeline()
    else:
        pipe, generator = None, None
    
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
        project_names_mixed.sort(key=lambda f: int(re.sub('\D', '', f)))
    except:
        pass
    for project_name_mixed in project_names_mixed:
        project_names.append(project_name_mixed)
       
    # run each project in PROJECTS_DIR in sequence
    for project_name in project_names:
        CURRENT_PROJECT_DIR = CWD+'/'+PROJECTS_DIR+'/'+project_name
        os.chdir(CURRENT_PROJECT_DIR)
        print("Current working directory: {0}".format(os.getcwd())) 
        
        if(Path(f"{CURRENT_PROJECT_DIR}/{project_name}.mp4").is_file() == False):
            if DEBUG == 'yes':
                pause()
                
            # Create directiories for text, audio, images and video files    
            createFolders()
            
            # load story and split it by sentence
            number_of_story_sentences = load_and_split_to_sentences("story.txt")
            
            # group sentences into story fragments of a given length
            number_of_story_fragments = sentences_to_fragments(number_of_story_sentences, FRAGMENT_LENGTH)
            
            image_prompts = []
            
            using_video_fragments_Processes = False
  
            # for each story fragment
            for i in range(number_of_story_fragments):
                print(f"{showTime()} {i} of {number_of_story_fragments-1}:")
                
                # vvvvvv pause / unpause

                # ^^^^^^ pause / unpause
                
                # vvvvvv significant speedup, but needs fast CPU and more than 32GB of RAM 
                if(SPEED_UP == 'yes'):
                    # generate prompts in advance if using keyBERT
                    if(USE_CHATGPT == 'no'):
                        # stay ahead of current iteration by this many steps 
                        steps_to_stay_ahead = 10
                        j = i + steps_to_stay_ahead
                        if(j < number_of_story_fragments):
                            if(Path(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{j}.txt").is_file() == False):
                                # translate fragment into prompt
                                print(f"{showTime()} {j} of {number_of_story_fragments-1} preparing prompts in advance")
                                test_thread = Process(target=fragment_toPrompt, args=(j, CURRENT_PROJECT_DIR))
                                test_thread.start()
                # ^^^^^^ significant speedup, but needs fast CPU and more than 32GB of RAM 
                
                # create voiceover using edge_tts
                if(Path(f"{CURRENT_PROJECT_DIR}/audio/voiceover{i}.wav").is_file() == False):
                    story_fragment = read_file(f"text/story_fragments/story_fragment{i}.txt")
                    do_it = True
                    while(do_it):
                        try: 
                            asyncio.get_event_loop().run_until_complete(create_vioceover(story_fragment, CURRENT_PROJECT_DIR))
                            do_it = False
                        except Exception as e:
                            wait_time = 10
                            print(f"Exception!!! \n{e} \nWaiting for {wait_time} seconds and trying again...")
                            time.sleep(wait_time)
                
                if(Path(f"{CURRENT_PROJECT_DIR}/text/image_prompts/image_prompt{i}.txt").is_file() == False):
                    # translate fragment into prompt
                    fragment_toPrompt(i, CURRENT_PROJECT_DIR)
                    
                if(Path(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").is_file() == True) and (Path(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4").is_file() == False):
                    # if cpu usage is more than 90%, wait for current loop to end (tweak this value based on your needs)
                    if (int(psutil.cpu_percent(1)) > 90):
                        # In case all images are ready, but none of videos are, video clip creation process will start all clips at once, eat all RAM and crash system. Add few seconds of delay between steps to prevent this. 
                        print('Main: High cpu usage -> Waiting for a few seconds before starting next loop...')
                        time.sleep(10)
                
                if(Path(f"{CURRENT_PROJECT_DIR}/images/image{i}.jpg").is_file() == False):
                    # generate image form prompt
                    prompt_to_image(pipe, generator, i, image_width, image_height, CURRENT_PROJECT_DIR)
                    
                if(Path(f"{CURRENT_PROJECT_DIR}/videos/video{i}.mp4").is_file() == False):
                    # create video clip using story fragment and generated image
                    # create a new process
                    using_video_fragments_Processes = True
                    process = Process(target = createVideoClip, args = (i, CURRENT_PROJECT_DIR))
                    # start the new process
                    process.start()
 
            if(using_video_fragments_Processes):
                # wait for the new process to finish
                print('Main: Waiting for video_fragments process to terminate...')
                # block until all tasks are done
                process.join()
                # continue on
                print('Main: video_fragments joined, continuing on')
            
            if(Path(CURRENT_PROJECT_DIR+'/'+project_name+".mp4").is_file() == False):
                # create final video
                final_video_process = Process(target = makeFinalVideo, args = (project_name, CURRENT_PROJECT_DIR))
                final_video_process.start()
                time.sleep(30)
                # if free virtual memory is less than 100GB, wait for final_video_process to end (tweak this value based on your needs)
                if (int(psutil.swap_memory()[2]/1000000000) < 100):
                    # block until all tasks are done
                    print('Main: High vmem usage -> Waiting for final_video_process process to terminate...')
                    final_video_process.join()
                    print('Main: final_video_process joined, continuing on')

        
   
   
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
