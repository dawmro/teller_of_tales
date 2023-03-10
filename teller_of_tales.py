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


# Use API_KEY imported from environment variables
openai.api_key = os.environ['OPENAI_TOKEN']

# show step by step debug info?
DEBUG = True

# minimal amount of words to put in each story fragment
FRAGMENT_LENGTH = 10

# select model to use
model_engine = "text-davinci-003"

# set parameters for image
lowmem = True
seed = -1
image_width = 848
image_height = 480


def write_list(a_list, filename):
    print("Started writing list data into a json file")
    with open(filename, "w") as fp:
        json.dump(a_list, fp)
        print("Done writing JSON data into .json file")


def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list
        
                
def write_file(file_content, filename):
    print("Started writing file_content data into a file")
    with open(filename, "w") as fp:
        fp.write(file_content)
        print("Done file_content data into a file")
        
        
def read_file(filename):
    with open(filename, "r", encoding='cp1252') as fp:
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
    story = story_raw.replace('???', '').replace('???', '').replace('??????', ' ')

    # split story into list of sentences
    story_sentences_list = sent_tokenize(story)
    
    # write_list(story_sentences_list, "text/story_sentences_list.json")
    
    for i, story_sentence in enumerate(story_sentences_list):
        write_file(story_sentence, f"text/story_sentences/story_sentence{i}.txt")
    
    if DEBUG:
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
            if DEBUG:
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
    
    if DEBUG:
        # display story enumerating through each sentence
        for i, story_fragment in enumerate(story_fragments):
            print( i, story_fragment)
        print("\n!!!!!!!!!!!!!!\nThis is last chance to make changes in story_fragments.txt files\n!!!!!!!!!!!!!!")
        pause()
        
    dir_path = r'text/story_fragments'
    number_of_files = len(fnmatch.filter(os.listdir(dir_path), 'story_fragment*.txt'))
    print('number_of_files:', number_of_files)
    
    return number_of_files
    
    
def prompt_to_image(i, image_width, image_height):
    do_it = True
    while(do_it):
        try:
            image_prompt = read_file(f"text/image_prompts/image_prompt{i}.txt")
            print(i, image_prompt)
            # clear cuda cache
            with torch.no_grad():
                torch.cuda.empty_cache() 

            possitive_prompt_sufix = " [(extremely detailed CG unity 8k wallpaper), nostalgia, ((professional majestic oil painting)), trending on ArtStation, trending on CGSociety, High Detail, Sharp focus, ((dramatic)), by midjourney, beautiful and detailed lighting, realism, shadows]"
         
            negative_prompt = "canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"
            
            model_id = "darkstorm2150/Protogen_Infinity_Official_Release"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            
            # if limited by GPU memory (4GB VRAM):
            if lowmem == True: 
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
            
            prompt = image_prompt + possitive_prompt_sufix
                
            image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=image_height, width=image_width, guidance_scale=7.5, generator=generator, num_inference_steps=25).images[0]

            image.save(f"images/image{i}.jpg")
            
            do_it = False
            
        except:
            print("Exception!!! From Hugginface probably, don't really care about details. \nWaiting for 60 seconds and trying again...")
            time.sleep(60)
            prompt_to_image(i, image_width, image_height)


async def create_vioceover(story_fragment) -> None:
    TEXT = story_fragment
    VOICE = "en-GB-SoniaNeural"
    # VOICE = "en-GB-RyanNeural"
    OUTPUT_FILE = f"audio/voiceover{i}.mp3"
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)
    
   
def createVideoClip(i):

    story_fragment = read_file(f"text/story_fragments/story_fragment{i}.txt")

    # load the audio file using moviepy
    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
    
    # add audio fadein / fadeout ot minimize sound glitches
    audio_clip = audio_clip.audio_fadein(0.05).audio_fadeout(0.05)
    
    # add 1 second silence to begining of audio 
    silence = AudioClip(make_frame = lambda t: 0, duration = 1.0)
    audio_clip = concatenate_audioclips([silence, audio_clip])
    
    # get audio duration
    audio_duration = audio_clip.duration
    
    # load the image file using moviepy
    image_clip = ImageClip(f"images/image{i}.jpg").set_duration(audio_duration)
    
    # use moviepy to create a text clip from the text
    screensize = (image_width, image_height)
    text_clip = TextClip(story_fragment, fontsize=int(0.0599*image_height), font="Impact", color="black", stroke_color="white", stroke_width=round(0.0026*image_height, 1), size=screensize, method='caption', align="South")
    text_clip = text_clip.set_duration(audio_duration)
    
    # concatenate the audio, image, and text clips
    clip = image_clip.set_audio(audio_clip)
    video = CompositeVideoClip([clip, text_clip])
    
    # save Video Clip to a file
    video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
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
            )
            do_it = False
            answer = completions.choices[0].text
        except:
            print("Exception!!! Waiting for 60 seconds and trying again...")
            time.sleep(60)
            answer = askChatGPT(text, model_engine)
    
    return answer  


def fragment_toPrompt(i):
    story_fragment = read_file(f"text/story_fragments/story_fragment{i}.txt")
    prefix = "Suggest good image to illustrate the following fragment from story, make descrpition short and precise, one sentence, max 10 words: "
    # translate fragment into prompt
    image_prompt = askChatGPT(prefix + story_fragment, model_engine).strip()
    print(i, story_fragment)
    print(i, image_prompt)
    write_file(image_prompt, f"text/image_prompts/image_prompt{i}.txt")     
    

def createListOfClips():

    clips = []
    l_files = os.listdir("videos")
    l_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for file in l_files:
        clip = VideoFileClip(f"videos/{file}")
        clips.append(clip)
    
    return clips


if __name__ == "__main__":

    print(f"{showTime()}")
    
    if DEBUG:
        pause()
        
    # Create directiories for text, audio, images and video files    
    createFolders()
    
    # load story and split it by sentence
    number_of_story_sentences = load_and_split_to_sentences("story.txt")
    
    # group sentences into story fragments of a given length
    number_of_story_fragments = sentences_to_fragments(number_of_story_sentences, FRAGMENT_LENGTH)
    
    # convert each story fragment into prompt and use it to generate image

    image_prompts = []
    
    # for each story fragment
    for i in range(number_of_story_fragments):
        print(f"{showTime()} {i} of {number_of_story_fragments-1}:")
        
        # create voiceover using edge_tts
        if(Path(f"audio/voiceover{i}.mp3").is_file() == False):
            story_fragment = read_file(f"text/story_fragments/story_fragment{i}.txt")
            asyncio.get_event_loop().run_until_complete(create_vioceover(story_fragment))
        
        if(Path(f"text/image_prompts/image_prompt{i}.txt").is_file() == False):
            # translate fragment into prompt
            fragment_toPrompt(i)
        
        if(Path(f"images/image{i}.jpg").is_file() == False):
            # generate image form prompt 
            prompt_to_image(i, image_width, image_height)
        
        if(Path(f"videos/video{i}.mp4").is_file() == False):
            # create video clip using story fragment and generated image
            # create a new thread
            thread = Thread(target=createVideoClip, args=[i])
            # start the new thread
            thread.start()
        
        # if DEBUG:
            # pause()
            
            
    # wait for the new thread to finish
    print('Main: Waiting for thread to terminate...')
    thread.join(timeout=30)
    # continue on
    print('Main: Continuing on')
    if thread.is_alive():
        print('Main: The target thread is still running')
    else:
        print('Main: The target thread has terminated')
    
    # create sorted list of clips
    print(f"{showTime()} Fixing order of video clips")
    clips = createListOfClips()
    
    # add audio fade to prevent audio glitches when combining multiple clips
    print(f"{showTime()} Adding audio fadein / fadeout...")
    clips = [clip.audio_fadein(0.10).audio_fadeout(0.10) for clip in clips]
    
    # add video fade to create smooth transitions
    print(f"{showTime()} Adding video fadein / faedout...")
    clips = [clip.crossfadein(1.0).crossfadeout(1.0) for clip in clips]
    
    # combine all clips into final video
    print(f"{showTime()} Concatenate all clips into final video...")
    final_video = concatenate_videoclips(clips, padding=-1, method="compose")
    final_video = final_video.write_videofile("final_video.mp4")
    print(f"{showTime()} Final video created successfully!")

        
   
   
'''
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

Name: en-US-SteffanNeural
Gender: Male

Name: en-ZA-LeahNeural
Gender: Female

Name: en-ZA-LukeNeural
Gender: Male
'''