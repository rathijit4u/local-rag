import spacy
import time

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
nlp.max_length = 10**7

def semantic_chunk(text: str, max_limit=25):
    chunks = []
    current_chunk = []
    doc = nlp(text)
    for sentence in doc.sents:
        current_chunk.append(sentence.text)
        if len(current_chunk) > max_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if len(current_chunk) > 0:
        chunks.append(" ".join(current_chunk))
    return chunks

if __name__ == '__main__':
    with open("data/01 Harry Potter and the Sorcerers Stone.txt", "r", encoding="utf-8") as file:
        book_text = file.read()
    start = time.time()
    text_chunks = semantic_chunk(book_text)
    for index, chunk in enumerate(text_chunks):
        with open(f"data/out/new_book_{index}.txt", "w", encoding="utf-8") as file:
            file.write(chunk)
    print("Time taken: ", time.time() - start)