# Move files processed with process_content.py to correct directories

import os
import shutil

# Directory containing your text files
base_directory = 'I Woke Up Piloting The Strongest Starship 2/text/c1-470'

# Range of file numbers
start_number = 1
end_number = 470

# Base name for new directories
base_name = "I Woke Up Piloting the Strongest Starship, part "

for number in range(start_number, end_number + 1):
    # Create a new directory
    new_directory = os.path.join(base_directory, base_name + str(number))
    os.makedirs(new_directory, exist_ok=True)
    
    # File name to process
    file_name = f"{number:05d}.txt"
    
    # File path to process
    file_path = os.path.join(base_directory, file_name)
    
    # Rename and move the file to the new directory
    new_file_path = os.path.join(new_directory, "story.txt")
    shutil.move(file_path, new_file_path)

print("Processing complete.")
