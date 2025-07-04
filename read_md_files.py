import os
from pathlib import Path
import markdown
from bs4 import BeautifulSoup

def read(file_name):
    file_path = Path(file_name)
    if file_path.suffix != ".md":
        raise ValueError("File name must end with .md extension")
    try:
        # Step 1: Read the Markdown file
        with open(file_name, "r", encoding="utf-8") as file:
            md_content = file.read()

        # Step 2: Convert Markdown to HTML
        html = markdown.markdown(md_content)

        # Step 3: Parse HTML and extract plain text
        soup = BeautifulSoup(html, "html.parser")
        plain_text = soup.get_text()

        #print(plain_text)
        return plain_text
    except FileNotFoundError:
        print(f"File '{file_name}' not found")
        raise

def get_all_files_under_dir(directory_path):
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = Path(file)
            if file_path.suffix == ".md":
                all_files.append(os.path.join(root, file))
    return all_files

def read_all_files_under_dir(directory_path):
    file_texts = []
    all_files = get_all_files_under_dir(directory_path)
    for file in all_files:
        file_texts.append(read(file))
    return file_texts

def write_to_file(file_text, file_name):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(file_text)
if __name__ == "__main__":
    #read(r"C:\Users\rjs\Documents\GitHub\cowj\manual\pauth.md")
    file_names = read_all_files_under_dir(r"C:\Users\rjs\Documents\GitHub\cowj\manual")
    write_to_file("\n".join(file_names), "data/md_output.txt")
