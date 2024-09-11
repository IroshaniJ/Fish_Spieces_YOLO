import os

def find_jpg_files(folder_path, output_file):
    # Open the output file to write the paths
    with open(output_file, 'w') as file:
        # Traverse the directory tree
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Check if the file is a .jpg file
                if filename.lower().endswith('.jpg'):
                    # Get the full path of the .jpg file
                    file_path = os.path.join(root, filename)
                    # Write the file path to the output file
                    file.write(file_path + '\n')

# Example usage
folder_path = './test'  # Replace with your folder path
output_file = 'test.txt'  # Output text file with the paths

find_jpg_files(folder_path, output_file)