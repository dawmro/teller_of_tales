from datetime import datetime

def showTime():
    return str("["+datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')+" UTC]")

if __name__ == "__main__":

    print(f"{showTime()}")