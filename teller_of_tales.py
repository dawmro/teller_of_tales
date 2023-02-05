# if first run then run installer 
import nltk
nltk.download()


from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime

import os



# show step by step debug info?
DEBUG = True



def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")
    


def pause():
    programPause = input("Press the <ENTER> key to continue...")

    
    
def createFolders():
    
    if not os.path.exists("text"):
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

    # remove quotes and weird symbols from story
    story = story_raw.replace('“', '').replace('”', '').replace('——', '').replace('‘', '').replace('’', '')

    # split story into list of sentences
    story_sentences_list = sent_tokenize(story)
    
    if DEBUG:
        # display story enumerating through each sentence
        for i, story_sentence in enumerate(story_sentences_list):
            print( i, story_sentence)
        pause()
    
    return story_sentences_list
    
    

if __name__ == "__main__":

    print(f"{showTime()}")
    
    if DEBUG:
        pause()
    
    # Create directiories for text, audio, images and video files    
    createFolders()
    
    # load story and split it by sentence
    story_sentences_list = load_and_split_to_sentences("story.txt")