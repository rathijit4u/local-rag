from MilvusUtil import MilvusUtil
from llm_util import get_text_from_llm
from semantic_chunking_util import semantic_chunk
import time
harry_collection_name = "harry_collection_BGEM3_2"

def insert_text_in_vector_db(text, collection_name="demo_collection"):
    chunks = semantic_chunk(text, max_limit=10)
    milvus_util = MilvusUtil()
    milvus_util.insert_docs_milvus_db(chunks, collection_name=collection_name)

def insert_file_text_in_vector_db(file_name, collection_name="demo_collection"):
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    insert_text_in_vector_db(text, collection_name=collection_name)

def find_answer(query):
    milvus_util = MilvusUtil()
    rerank_results = milvus_util.search_rerank(query,harry_collection_name, 3)
    #[print(f"result={result}") for result in rerank_results]
    context = "\n".join(rerank_results)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    print(prompt)
    return get_text_from_llm(prompt)


if __name__ == '__main__':
    start = time.time()
    books = []
    for book in books:
        #insert_file_text_in_vector_db(f"data/{book}", harry_collection_name)
        print(f"{book} is done")
    print(find_answer("Whose identity does Harry adopt to infiltrate the Ministry of Magic in Harry Potter and the Deathly Hallows?"))
    print("Time taken: ", time.time() - start)