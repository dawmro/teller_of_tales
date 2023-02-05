from datetime import datetime

import os


def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")
    
    
    
def createFolders():
    
    if not os.path.exists("text"):
        os.makedirs("text") 
    if not os.path.exists("audio"):
        os.makedirs("audio")         
    if not os.path.exists("images"):
        os.makedirs("images")     
    if not os.path.exists("videos"):
        os.makedirs("videos")
    
    

if __name__ == "__main__":

    print(f"{showTime()}")
    
    # Create directiories for text, audio, images and video files    
    createFolders()