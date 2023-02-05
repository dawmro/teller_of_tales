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


# Use API_KEY imported from environment variables
openai.api_key = os.environ['OPENAI_TOKEN']

# show step by step debug info?
DEBUG = True

# minimal amount of words to put in each story fragment
FRAGMENT_LENGTH = 10

# select model to use
model_engine = "text-davinci-003"
   


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
    

def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")


def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
    
def createFolders():
    
    if not os.path.exists("text"):
        print("no path")
        os.makedirs("text") 
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

    # remove quotes and other characters from story
    story = story_raw.replace('“', '').replace('”', '').replace('‘', '').replace('’', '').replace('——', '').replace('\n', ' ')

    # split story into list of sentences
    story_sentences_list = sent_tokenize(story)
    
    write_list(story_sentences_list, "text/story_sentences_list.json")
    
    if DEBUG:
        # display story enumerating through each sentence
        for i, story_sentence in enumerate(story_sentences_list):
            print( i, story_sentence)
        print("\n!!!!!!!!!!!!!!\nThis is last chance to make changes in story_sentences_list.json file\n!!!!!!!!!!!!!!")
        pause()
        
    story_sentences_list = read_list("text/story_sentences_list.json")
    
    return story_sentences_list


def sentences_to_fragments(story_sentences_list, FRAGMENT_LENGTH):

    # story divided into fragments
    story_fragments = []

    # fragment currently being worked on
    current_fragment = None

    # current fragment word counter
    current_fragment_word_counter = 0

    # for every sentence in list of sentences
    # combine sentences form story into fragments
    for story_sentence in story_sentences_list:

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
    
    write_list(story_fragments, "text/story_fragments.json")
    
    if DEBUG:
        # display story enumerating through each sentence
        for i, story_fragment in enumerate(story_fragments):
            print( i, story_fragment)
        print("\n!!!!!!!!!!!!!!\nThis is last chance to make changes in story_fragments.json file\n!!!!!!!!!!!!!!")
        pause()
        
    story_fragments = read_list("text/story_fragments.json")
    
    return story_fragments
    
    
def prompt_to_image(i, image_prompt, image_width, image_height):

    # clear cuda cache
    with torch.no_grad():
        torch.cuda.empty_cache() 
    
    # set parameters for image 
    seed = 1337

    possitive_prompt_sufix = " (extremely detailed CG unity 8k wallpaper), nostalgia, professional majestic oil painting, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, by midjourney and greg rutkowski, realism, beautiful and detailed lighting, shadows, by Jeremy Lipking"

    negative_prompt = "disfigured, kitsch, ugly, oversaturated, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, ugly, disgusting, poorly drawn, childish, mutilated, mangled, old, surreal, text"
    
    model_id = "darkstorm2150/Protogen_v2.2_Official_Release"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(seed)
    
    # consider chunking the attention computation if limited by GPU memory 
    pipe.enable_attention_slicing()
    
    # uncomment to disable NSFW filter
    # def dummy_checker(images, **kwargs): return images, False
    # pipe.safety_checker = dummy_checker
    
    prompt = image_prompt + possitive_prompt_sufix
        
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, height=image_height, width=image_width, guidance_scale=7.5, generator=generator, num_inference_steps=25).images[0]

    image.save(f"images/image{i}.jpg")
    
    
def createVideoClip(i, story_fragment):

    # create gTTS instance and save to a file
    tts = gTTS(text=story_fragment, lang='en', slow=False)
    tts.save(f"audio/voiceover{i}.mp3")
    
    # load the audio file using moviepy
    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
    audio_duration = audio_clip.duration
    
    # load the image file using moviepy
    image_clip = ImageClip(f"images/image{i}.jpg").set_duration(audio_duration)
    
    # use moviepy to create a text clip from the text
    screensize = (image_width, image_height)
    text_clip = TextClip(story_fragment, fontsize=35, font="Impact", color="black", stroke_color="white", stroke_width=1.5, size=screensize, method='caption', align="South")
    text_clip = text_clip.set_duration(audio_duration)
    
    # concatenate the audio, image, and text clips
    clip = image_clip.set_audio(audio_clip)
    video = CompositeVideoClip([clip, text_clip])
    
    # save Video Clip to a file
    video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
    print(f"{showTime()} The Video{i} Has Been Created Successfully!")
        
    
def askChatGPT(text, model_engine):

    completions = openai.Completion.create(
        engine=model_engine,
        prompt=text,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.9,
    )
    return completions.choices[0].text
    

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
    story_sentences_list = load_and_split_to_sentences("story.txt")
    
    # group sentences into story fragments of a given length
    story_fragments = sentences_to_fragments(story_sentences_list, FRAGMENT_LENGTH)
    
    # convert each story fragment into prompt and use it to generate image
    image_width = 1024
    image_height = 576 
    
    image_prompts = []
    
    # for each story fragment
    for i, story_fragment in enumerate(story_fragments):
        print(f"{showTime()}")
        prefix = "Suggest good image to illustrate the following fragment from story, make descrpition short and precise, one sentence, max 10 words: "
        # translate fragment into prompt 
        try:
            image_prompt = askChatGPT(prefix + story_fragment, model_engine).strip()
            print(i, image_prompt)
            image_prompts.append(image_prompt)
            write_list(image_prompts, "text/image_prompts.json")
            image_prompts = read_list("text/image_prompts.json")
        except:
            print(f"{showTime()} Cannot connect with OpenAI servers. \nProbable cause: No Internet connection, Invalid API token, Too much calls in short time")
            exit()
        
        # generate image form prompt 
        prompt_to_image(i, image_prompt, image_width, image_height)
        
        # create video clip using story fragment and generated image
        createVideoClip(i, story_fragment)
        
        # if DEBUG:
            # pause()
    
    # create sorted list of clips
    print(f"{showTime()} Fixing order of video clips")
    clips = createListOfClips()
    
    # add audio fade to prevent audio glitches when combining multiple clips
    clips = [clip.audio_fadein(0.04).audio_fadeout(0.04) for clip in clips]
    
    # combine all clips into final video
    print(f"{showTime()} Concatenate all clips into final video...")
    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.write_videofile("final_video.mp4")
    print(f"{showTime()} Final video created successfully!")

        
   