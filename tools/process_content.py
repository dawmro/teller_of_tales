# Process content of files downloaded via Lightnovel Crawler https://github.com/dipu-bd/lightnovel-crawler

import os
import re

# Directory containing your text files
directory = 'I Woke Up Piloting The Strongest Starship 2/text/c1-470'

# List of files to process
files = [f for f in os.listdir(directory) if f.endswith('.txt')]

# Regular expression pattern to match unwanted lines
pattern = r'^\[previous_page\]|\[next_page\]|^Translator:|^Editor:'

# Lines to add at the beginning of each file
new_lines = ["I Woke Up Piloting the Strongest Starship, so I Became a Space Mercenary,\n", "Written by Ryuto,\n"]


for file in files:
    file_path = os.path.join(directory, file)
    
    # Read the file and remove unwanted lines
    with open(file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()[1:] # 1
        #lines = infile.readlines() # 2+

    # Remove empty lines
    lines = [line for line in lines if line.strip()] # 1+
    
    # Add a period at the end of the first line 
    if lines: # 1
        lines[0] = lines[0].rstrip('\r\n') + '.\n\n' # 1

    # Add the new lines at the beginning
    lines = new_lines + lines # 1

    for line in lines:
        line = line.replace('\r', '\n')
        
 

    # Filter out lines that match the pattern
    filtered_lines = [line for line in lines if not re.match(pattern, line)]

    # Find and display the problematic character
    for line in filtered_lines:
        for char in line:
            try:
                char.encode('utf-8')
            except UnicodeEncodeError:
                print(f"Problematic character: {char}")
                break
        else:
            continue
        break

    # Write the filtered lines back to the file with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(filtered_lines)

print("Processing complete.")




