# if first run then run installer 
import nltk
nltk.download()


from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime

import os
import json


# show step by step debug info?
DEBUG = True



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

    # remove quotes from story
    story = story_raw.replace('“', '').replace('”', '').replace('——', '')

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
    
    

if __name__ == "__main__":

    print(f"{showTime()}")
    
    if DEBUG:
        pause()
    
    # Create directiories for text, audio, images and video files    
    createFolders()
    
    # load story and split it by sentence
    story_sentences_list = load_and_split_to_sentences("story.txt")