# if first run then run installer 
import nltk
nltk.download()


from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime

import os
import json


# show step by step debug info?
DEBUG = True

# minimal amount of words to put in each story fragment
FRAGMENT_LENGTH = 10



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